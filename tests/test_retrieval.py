import tempfile
import unittest
from pathlib import Path
import json

from retrieval import Chunk, Retriever, chunks_from_markdown, load_corpus, split_text


class RetrievalTests(unittest.TestCase):
    def test_split_text_preserves_content(self):
        chunks = split_text("Force causes acceleration. " * 80, max_chars=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertIn("Force causes acceleration.", " ".join(chunks))
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_retrieval_returns_matching_page_citation(self):
        passages = [
            Chunk("Force equals mass times acceleration.", "physics.pdf", "Page 17",
                  "p1", "Board X", "11", "Physics", "PDF"),
            Chunk("Plants convert sunlight into chemical energy.", "biology.pdf", "Page 5",
                  "b1", "Board X", "11", "Biology", "PDF"),
        ]
        result = Retriever(passages).search("How are force mass and acceleration related?", limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk.citation, "Page 17")

    def test_markdown_citation_uses_heading_not_fake_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chapter.md"
            path.write_text("# Motion\n\nVelocity is displacement per time.\n", encoding="utf-8")
            chunks = chunks_from_markdown(path)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.citation == "Section: Motion" for chunk in chunks))

    def test_markdown_mathml_is_rendered_as_readable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chapter.md"
            path.write_text(
                '# Motion\nVelocity is <math><semantics><mrow><mi>v</mi><mo>=</mo>'
                '<mfrac><mrow><mi>Δx</mi></mrow><mrow><mi>Δt</mi></mrow></mfrac>'
                '</mrow><annotation>internal math data</annotation></semantics></math>.',
                encoding="utf-8",
            )
            chunks = chunks_from_markdown(path)
        self.assertIn("(Δx) / (Δt)", chunks[0].text)
        self.assertNotIn("<math", chunks[0].text)
        self.assertNotIn("internal math data", chunks[0].text)

    def test_blank_query_is_reported(self):
        retriever = Retriever([Chunk("A useful passage.", "book.pdf", "Page 1",
                                     "id", "Board", "11", "Physics", "PDF")])
        with self.assertRaises(ValueError):
            retriever.search("  ")

    def test_processed_dataset_keeps_page_and_chapter_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "processed"
            output.mkdir(parents=True)
            record = {
                "id": "physics:p3:c1", "book_id": "physics", "book_title": "Physics",
                "board": "Board", "class_level": "11", "subject": "Physics",
                "chapter": "Chapter 1: Motion", "page_number": 3,
                "text": "Velocity is displacement per unit time.", "chunk_id": 1,
                "source_file": "data/physics.pdf", "source_type": "PDF OCR",
            }
            (output / "passages.jsonl").write_text(json.dumps(record), encoding="utf-8")
            corpus = load_corpus(root)
        self.assertEqual(len(corpus), 1)
        self.assertEqual(corpus[0].citation, "Page 3")
        self.assertEqual(corpus[0].chapter, "Chapter 1: Motion")


if __name__ == "__main__":
    unittest.main()
