"""Telephony degradations for explicit robustness experiments, not codec authenticity claims."""
import numpy as np
from scipy.signal import butter, sosfilt

class TelephonyDegradationPipeline:
    def __init__(self, sr=16000): self.sr=sr
    def encode_decode_g711_mulaw(self, audio, sr=16000):
        from api.telephony import decode_mulaw_byte_chunk
        pcm=(np.clip(audio,-1,1)*32767).astype(np.int32)
        sign=np.where(pcm<0,0x80,0)
        mag=np.minimum(np.abs(pcm),32635)+132
        exponent=np.clip(np.floor(np.log2(np.maximum(mag,1))).astype(int)-7,0,7)
        mantissa=(mag>>(exponent+3))&15
        encoded=(~(sign|(exponent<<4)|mantissa)&255).astype(np.uint8)
        return decode_mulaw_byte_chunk(encoded.tobytes())
    def apply_amr_narrowband_filter(self,audio,sr=16000):
        # Bandwidth approximation only; this is not an AMR encoder.
        return sosfilt(butter(5,[300,3400],fs=self.sr,btype='bandpass',output='sos'),audio).astype(np.float32)
    def inject_environmental_noise(self,audio,snr_db=12.0):
        rng=np.random.default_rng(42)
        noise=rng.normal(size=len(audio))
        level=np.sqrt(np.mean(audio**2))/10**(snr_db/20)
        return np.clip(audio+noise*level,-1,1).astype(np.float32)
    def apply_realistic_phone_call_pipeline(self,audio):
        return self.encode_decode_g711_mulaw(self.inject_environmental_noise(self.apply_amr_narrowband_filter(audio)))

    def apply_voip_packet_loss(self,audio,sr=16000,packet_loss_rate=.05):
        output=audio.copy()
        rng=np.random.default_rng(42)
        frame=max(1,int(sr*.02))
        for start in range(0,len(output),frame):
            if rng.random()<packet_loss_rate: output[start:start+frame]=0
        return output
