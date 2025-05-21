import asyncio
import time
import numpy as np
import resampy
import textwrap
from frame_msg import FrameMsg, RxAudio, TxCode
from faster_whisper import WhisperModel

# ------------ Config ---------------------------------------------------
MODEL_SIZE  = "base"     # tiny / base / small …
STEP_SEC    = 1.0        # seconds between partial decodes
RATE_IN     = 8000       # Frame mic
RATE_OUT    = 16000      # Whisper expects
CONTEXT_SEC = 25         # rolling window on which we run Whisper
# -----------------------------------------------------------------------

def pcm16_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

async def main():
    """
    Capture audio from Frame, transcribe it using faster-whisper, and display the transcription
    """
    frame = FrameMsg()
    rx_audio = None
    
    try:
        # Load Whisper model
        print("Loading faster-whisper model...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("Whisper model loaded")
        
        print("Connecting to Frame...")
        await frame.connect()
        print("Connected successfully!")
        
        # Attach print response handler to see Frame's Lua print statements
        frame.attach_print_response_handler()
        
        # Create a custom Lua app that handles display and audio
        custom_lua_app = """
        local data = require('data.min')
        local code = require('code.min')
        local audio = require('audio.min')
        
        -- Message codes
        local AUDIO_CONTROL_MSG = 0x30
        local TEXT_UPDATE_MSG = 0x31
        
        -- Register message parsers
        data.parsers[AUDIO_CONTROL_MSG] = code.parse_code
        data.parsers[TEXT_UPDATE_MSG] = function(bytes)
            local str = string.sub(bytes, 1)
            return {text = str}
        end
        
        -- Main app function
        function app_loop()
            -- Initial display
            frame.display.text("Frame Transcription App", 10, 10)
            frame.display.text("Ready - Press Enter to start", 10, 50)
            frame.display.show()
            
            -- Tell host the app is running
            print("Frame app is running")
            
            local streaming = false
            
            while true do
                rc, err = pcall(
                    function()
                        -- Process incoming messages
                        local items_ready = data.process_raw_items()
                        
                        if items_ready > 0 then
                            -- Handle audio control messages
                            if data.app_data[AUDIO_CONTROL_MSG] ~= nil then
                                if data.app_data[AUDIO_CONTROL_MSG].value == 1 then
                                    -- Start audio
                                    streaming = true
                                    audio.start({sample_rate=8000, bit_depth=16})
                                    frame.display.text("Recording...", 10, 90)
                                    frame.display.show()
                                else
                                    -- Stop audio
                                    audio.stop()
                                    frame.display.text("Processing...", 10, 90)
                                    frame.display.show()
                                end
                                data.app_data[AUDIO_CONTROL_MSG] = nil
                            end
                            
                            -- Handle text update messages
                            if data.app_data[TEXT_UPDATE_MSG] ~= nil and data.app_data[TEXT_UPDATE_MSG].text ~= nil then
                                -- Clear display
                                frame.display.text("                                        ", 10, 10)
                                frame.display.text("                                        ", 10, 50)
                                frame.display.text("                                        ", 10, 90)
                                frame.display.text("                                        ", 10, 130)
                                frame.display.text("                                        ", 10, 170)
                                frame.display.text("                                        ", 10, 210)
                                frame.display.text("                                        ", 10, 250)
                                
                                -- Display new text
                                local text = data.app_data[TEXT_UPDATE_MSG].text
                                frame.display.text("Transcription:", 10, 10)
                                
                                -- Split text into lines (simple approach)
                                local y = 50
                                local start = 1
                                local line_length = 40
                                
                                while start <= #text do
                                    local display_text = string.sub(text, start, start + line_length - 1)
                                    frame.display.text(display_text, 10, y)
                                    y = y + 40
                                    start = start + line_length
                                    
                                    -- Prevent too many lines
                                    if y > 250 then break end
                                end
                                
                                frame.display.show()
                                data.app_data[TEXT_UPDATE_MSG] = nil
                            end
                        end
                        
                        -- Send audio data if streaming
                        if streaming then
                            local sent = audio.read_and_send_audio()
                            if sent == nil then
                                streaming = false
                            end
                        end
                        
                        -- Sleep to prevent CPU overuse
                        frame.sleep(0.01)
                    end
                )
                
                -- Handle errors
                if rc == false then
                    print("Error: " .. err)
                    break
                end
            end
        end
        
        -- Start the app
        app_loop()
        """
        
        # Upload our custom Lua app
        print("Uploading custom Lua app...")
        await frame.upload_file_from_string(custom_lua_app, "custom_transcription_app.lua")
        
        # Upload necessary Lua libraries
        print("Uploading libraries...")
        await frame.upload_stdlua_libs(lib_names=["data", "code", "audio"])
        
        # Start our custom Frame app
        print("Starting custom Frame app...")
        await frame.send_lua("require('custom_transcription_app')", await_print=True)
        
        # Wait a moment for the app to initialize
        await asyncio.sleep(2)
        
        # Set up audio receiver
        rx_audio = RxAudio(streaming=True)
        audio_queue = await rx_audio.attach(frame)
        
        # Initialize audio buffers
        buf_pcm8k = bytearray()
        buf_pcm16k = np.empty(0, dtype=np.float32)
        last_terminal = ""
        
        # Main loop
        while True:
            # Wait for user input to start recording
            await asyncio.to_thread(input, "Press Enter to start recording: ")
            
            # Clear buffers
            buf_pcm8k.clear()
            buf_pcm16k = np.empty(0, dtype=np.float32)
            last_terminal = ""
            
            # Start audio recording
            print("Starting recording...")
            await frame.send_message(0x30, TxCode(value=1).pack())
            
            # Send initial message to Frame
            await frame.send_message(0x31, "Listening...".encode())
            
            # Record and process audio
            print("Recording... (Press Enter to stop)")
            
            # Start a task to wait for user input to stop recording
            stop_event = asyncio.Event()
            stop_task = asyncio.create_task(wait_for_stop(stop_event))
            
            last_t = time.time()
            max_samples = CONTEXT_SEC * RATE_OUT
            
            try:
                while not stop_event.is_set():
                    try:
                        # Get audio packet with timeout
                        pkt = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        if pkt is None:  # Stream ended
                            break
                        buf_pcm8k += pkt
                        
                        # Process audio at regular intervals
                        if time.time() - last_t >= STEP_SEC:
                            last_t = time.time()
                            pcm_f32 = pcm16_to_f32(bytes(buf_pcm8k))
                            buf_pcm8k.clear()
                            if pcm_f32.size == 0:
                                continue
                            
                            # Resample 8 kHz → 16 kHz
                            up = resampy.resample(pcm_f32, RATE_IN, RATE_OUT)
                            buf_pcm16k = np.concatenate([buf_pcm16k, up])
                            if buf_pcm16k.size > max_samples:
                                buf_pcm16k = buf_pcm16k[-max_samples:]
                            
                            # Transcribe audio
                            segments, _ = model.transcribe(
                                buf_pcm16k,
                                language="en",
                                beam_size=5,
                                vad_filter=True,
                                word_timestamps=False,
                            )
                            
                            # Collect transcription
                            text = "".join(s.text for s in segments).strip()
                            if text and text != last_terminal:
                                # Print to terminal
                                print(f"Transcription: {text}")
                                last_terminal = text
                                
                                # Send to Frame
                                await frame.send_message(0x31, text.encode())
                    
                    except asyncio.TimeoutError:
                        # No audio packet received, continue
                        continue
            
            finally:
                # Cancel the stop task if it's still running
                if not stop_task.done():
                    stop_task.cancel()
                    try:
                        await stop_task
                    except asyncio.CancelledError:
                        pass
            
            # Stop recording
            print("Stopping recording...")
            await frame.send_message(0x30, TxCode(value=0).pack())
            
            # Final transcription
            if buf_pcm16k.size > 0:
                print("Processing final transcription...")
                await frame.send_message(0x31, "Processing final transcription...".encode())
                
                segments, _ = model.transcribe(
                    buf_pcm16k,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=False,
                )
                
                # Collect final transcription
                final_text = "".join(s.text for s in segments).strip()
                if final_text:
                    print(f"Final transcription: {final_text}")
                    await frame.send_message(0x31, final_text.encode())
                else:
                    await frame.send_message(0x31, "No speech detected.".encode())
            
            print("Ready for next recording.")
            
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if rx_audio and frame:
            rx_audio.detach(frame)
        if frame:
            frame.detach_print_response_handler()
            await frame.stop_frame_app()
            await frame.disconnect()
            print("Disconnected from Frame")

async def wait_for_stop(stop_event):
    """Wait for user input to stop recording"""
    await asyncio.to_thread(input, "")
    stop_event.set()

if __name__ == "__main__":
    asyncio.run(main())