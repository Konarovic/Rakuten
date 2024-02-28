from sklearn.pipeline import Pipeline
from src.features.text.transformers.cleaners import HtmlCleaner, LxmlCleaner, UrlCleaner, FileNameCleaner, BadHTMLCleaner, SpaceBeforeAdder, SpaceAroundAdder

class CleanTextPipeline(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ("HTML_Cleaner", HtmlCleaner()),
                ("LXML_Cleaner", LxmlCleaner()),
                ("URL_Cleaner", UrlCleaner()),
                ("File_Name_Cleaner", FileNameCleaner()),
                ("Bad_HTML_Cleaner", BadHTMLCleaner()),
                ("Space_Before_Adder", SpaceBeforeAdder())
            ]
        )

