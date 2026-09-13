import numpy as np
from core.forensic_pipeline import LanguageDetector, SR

def detector_without_model():
    detector=LanguageDetector.__new__(LanguageDetector)
    detector.model=None
    return detector

def test_no_model_does_not_invent_hindi_from_tone():
    audio=(.15*np.sin(2*np.pi*180*np.arange(SR*3)/SR)).astype(np.float32)
    result=detector_without_model().detect(audio)
    assert result['label']=='Unknown'
    assert result['languages']==[]

def test_interface_hint_is_not_language_evidence():
    audio=np.random.default_rng(17).normal(0,.05,SR*3).astype(np.float32)
    result=detector_without_model().detect(audio,language_hint='hi')
    assert result['label']=='Unknown'
    assert result['status']=='unavailable'

def test_audio_model_overrides_wrong_transcript_and_hint():
    class EnglishModel:
        def detect_language(self,audio): return 'en',.97,[('en',.97),('hi',.02),('bn',.01)]
    detector=detector_without_model();detector.model=EnglishModel()
    result=detector.detect(np.ones(SR*4,dtype=np.float32)*.1+np.random.default_rng(1).normal(0,.01,SR*4),transcript='नमस्ते',language_hint='hi')
    assert result['label']=='en'

def test_unknown_model_response_is_not_forced_to_language():
    class UncertainModel:
        def detect_language(self,audio): return 'hi',.3,[('hi',.3),('en',.29),('bn',.28)]
    detector=detector_without_model();detector.model=UncertainModel()
    result=detector.detect(np.random.default_rng(2).normal(0,.1,SR*4),language_hint='hi')
    assert result['label']=='Unknown'


def test_overlap_is_subtracted_from_individual_evidence():
    from core.speech_backends import exclusive_turns
    turns=[{'speaker':'A','start':0.,'end':5.}, {'speaker':'B','start':2.,'end':3.}]
    clean=exclusive_turns([turns[0]],turns)
    assert [(t['start'],t['end']) for t in clean]==[(0.,2.),(3.,5.)]
    assert exclusive_turns([turns[1]],turns)==[]

def test_three_simultaneous_voices_keep_count_but_no_isolated_audio():
    from core.speech_backends import exclusive_turns
    from core.forensic_pipeline import Diarization
    turns=[{'speaker':name,'start':0.,'end':4.} for name in ['A','B','C']]
    result=Diarization.finish(turns,'test','estimated',True,[])
    assert result['num_speakers']==3
    assert all(t['overlap'] for t in result['speaker_turns'])
    for turn in turns: assert exclusive_turns([turn],turns)==[]
