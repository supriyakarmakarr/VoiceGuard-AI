"""Compatibility risk engine. Indices are exploratory, never authenticity probabilities."""
import numpy as np

class RiskEngine:
    def __init__(self, weight_deep=.5, weight_baseline=.3, weight_forensic=.2, low_threshold=30., high_threshold=70., calibrator=None, ensemble=None):
        if not 0 <= low_threshold < high_threshold <= 100: raise ValueError('Invalid risk thresholds')
        self.weight_deep,self.weight_baseline,self.weight_forensic=weight_deep,weight_baseline,weight_forensic
        self.low_threshold,self.high_threshold=low_threshold,high_threshold
        self.calibrator,self.ensemble=calibrator,ensemble

    def evaluate(self,deep_result,baseline_result,forensic_signals,audio_duration=4.,raw_audio=None,sr=16000):
        available=bool(deep_result and baseline_result and forensic_signals and audio_duration>=2)
        d=deep_result.get('synthetic_probability') if deep_result else None
        b=baseline_result.get('synthetic_probability') if baseline_result else None
        available=available and d is not None and b is not None
        dsp=float(np.mean([forensic_signals.get(k,0) for k in ('hf_anomaly_score','prosody_anomaly_score','spectral_cutoff_score')]))
        if available and (not np.isfinite([d,b,dsp]).all() or abs(d-b)>.45): available=False
        score=round(float(np.clip(100*(self.weight_deep*d+self.weight_baseline*b+self.weight_forensic*dsp),0,100)),1) if available else None
        level='INSUFFICIENT_EVIDENCE' if score is None else 'LOW' if score<self.low_threshold else 'MEDIUM' if score<self.high_threshold else 'HIGH'
        text='Insufficient evidence' if score is None else f'{level.title()} acoustic concern; review required'
        return {'risk_score':score,'risk_level':level,'risk_color':{'HIGH':'#a84d53','MEDIUM':'#96702a','LOW':'#3d796c'}.get(level,'#7991ad'),
                'confidence_score':None,'confidence_label':'Low-confidence analysis' if score is not None else 'Insufficient evidence',
                'synthetic_probability':None,'genuine_probability':None,'tier_verdict':'UNCERTAIN','tier_badge':'REVIEW REQUIRED',
                'calibration':{'mode':'unvalidated','reason':'No deployment-domain validation set is supplied.'},
                'verdict':{'en':text,'hi':text,'bn':text},'indicators':[],
                'advisory':{'title':'Independent verification required','recommendation':'Confirm sensitive requests through a trusted second channel.','action':'Do not authorize a transaction based on this analysis.'}}

    def find_suspicious_intervals(self,audio,sr=16000):
        # Legacy API: temporal scoring is performed by ForensicPipeline on original-timeline segments.
        return []
