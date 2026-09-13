"""Legacy ensemble adapter. Only a successfully loaded trained model may contribute.
WavLM/AASIST/RawNet placeholders were untrained and are intentionally not instantiated.
The current API uses ForensicPipeline for CNN + acoustic ML + measured DSP fusion.
"""
from pathlib import Path
from core.deep_learning_model import DeepLearningVoiceClassifier

class MultiModelEnsemble:
    def __init__(self, deep_cnn_path=None, device='cpu'):
        self.model=None
        if deep_cnn_path and Path(deep_cnn_path).is_file():
            try: self.model=DeepLearningVoiceClassifier(deep_cnn_path)
            except Exception: pass

    def predict_speaker_audio(self,audio,sr=16000):
        score=None
        if self.model is not None and len(audio)/sr>=2:
            score=round(self.model.predict(audio)['synthetic_probability']*100,1)
        return {'synthetic_probability':None,'genuine_probability':None,'risk_score':score,'confidence':None,
                'tier_verdict':'UNCERTAIN','verdict_badge':'REVIEW REQUIRED','verdict_desc':'Uncalibrated trained-model response; not proof of authenticity.',
                'agreement_score':None,'model_scores':{'spec_cnn':score} if score is not None else {}}
