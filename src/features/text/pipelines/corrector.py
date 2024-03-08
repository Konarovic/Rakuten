from sklearn.pipeline import Pipeline
from src.features.text.transformers.encoding_corrector import EncodingIsolator, ExcessPunctuationRemover, HandMadeCorrector, EncodingApostropheCorrector,\
      ClassicPatternsCorrector, PronounsApostropheCorrector


class CleanEncodingPipeline(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ("Isolate Special duplicates", EncodingIsolator()),
                ("Remove Excess Punctuation", ExcessPunctuationRemover()),
                ("Correct Apostrophes", EncodingApostropheCorrector()),
                ("Correct QU Apostrophes", PronounsApostropheCorrector())
                #("Hand Made Correction", HandMadeCorrector()),
                #("Correct Classic patterns", ClassicPatternsCorrector())

                
                
            ]
        )

