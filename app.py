from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
from fastapi.responses import FileResponse

app = FastAPI()

# Initialize text-to-audio pipeline
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

# Initialize Musicgen model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

@app.get("/generate_music/{style}")
async def generate_music(style: str):
    try:
        # Generate audio from text
        music = synthesiser(style, forward_params={"do_sample": True})
        
        # Generate music from the specified style
        inputs = processor(
            text=[style],
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs, max_new_tokens=256)

        # Write generated audio to WAV file
        wav_filename = "musicgen_out.wav"
        scipy.io.wavfile.write(wav_filename, rate=music["sampling_rate"], data=music["audio"])

        # Return the generated audio as a file download response
        return FileResponse(wav_filename, media_type="audio/wav", filename="generated_music.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

