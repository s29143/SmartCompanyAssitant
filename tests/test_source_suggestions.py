import unittest

from langchain_core.documents import Document

from src.rag_chain import build_source_recommendations


class BuildSourceRecommendationsTests(unittest.TestCase):
    def test_build_source_recommendations_selects_up_to_three_links(self):
        docs = [
            Document(page_content="one", metadata={"source": "about-us", "type": "wordpress"}),
            Document(page_content="two", metadata={"source": "services", "type": "wordpress"}),
            Document(page_content="three", metadata={"source": "pricing", "type": "wordpress"}),
            Document(page_content="four", metadata={"source": "contact", "type": "wordpress"}),
        ]

        recommendations = build_source_recommendations(docs)

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(
            [item["source"] for item in recommendations],
            ["about-us", "services", "pricing"],
        )
        self.assertEqual(recommendations[0]["url"], "/about-us/")


if __name__ == "__main__":
    unittest.main()
