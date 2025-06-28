import spacy
from spacy.tokens import DocBin
import json

nlp = spacy.blank("en")  # blank English pipeline
doc_bin = DocBin()

with open("annotations/train_data.json", "r", encoding="utf8") as f:
    for line in f:
        data = json.loads(line)
        text = data["text"]
        entities = data["entities"]

        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label)
            if span is None:
                print(f"Skipping entity ({start}, {end}, {label}) due to misalignment.")
            else:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)

doc_bin.to_disk("train_data.spacy")
