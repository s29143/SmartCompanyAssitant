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

    def test_build_source_recommendations_filters_irrelevant_sources(self):
        docs = [
            Document(page_content="O nas i historii firmy", metadata={"source": "about-us", "type": "wordpress"}),
            Document(page_content="Cennik i ceny usług", metadata={"source": "pricing", "type": "wordpress"}),
            Document(page_content="Kontakt z nami", metadata={"source": "contact", "type": "wordpress"}),
        ]

        recommendations = build_source_recommendations(docs, question="Jakie są ceny usług?")

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["source"], "pricing")
        self.assertEqual(recommendations[0]["url"], "/pricing/")


if __name__ == "__main__":
    unittest.main()
