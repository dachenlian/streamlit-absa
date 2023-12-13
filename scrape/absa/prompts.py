from string import Template

from scrape.absa import aspects


_ABSA_OUTPUT_FORMAT = """{'text_1': {'aspect_1': 'positive', 'aspect_2': 'negative'}, 'text_2': {...}, ... }"""

GET_ABSA_MOVIE_PROMPT = Template(
    "\n".join(
        [
            "The following is a list of movie reviews:\n",
            "${text}",
            "END OF REVIEWS.\n\n"
            "Perform aspect-based sentiment analysis (positive, neutral, negative) on the above texts.\n\n",
            "Include an 'overall' aspect that describes the overall sentiment of the review.\n\n",
            "Furthermore, include a 'contains_spoilers' key with a boolean value indicating whether the review contains spoilers.\n\n",
            "If applicable to the review, choose aspects from the following:",
            f"{', '.join(aspects.MOVIE_ASPECTS)}\n\n"
            "Also include more personal aspects that may be helpful for potential viewers, such as the performance of a particular actor.\n\n",
            "Return a nested JSON object with the outer key being the index of the review, and the value being another JSON object with the aspect as the key and the sentiment as the value:\n",
            "The output should look like this:\n",
            _ABSA_OUTPUT_FORMAT,
        ]
    )
)

GET_ABSA_FINANCE_PROMPT = Template(
    "\n".join(
        [
            "The following is a post on a discussion forum and its comments:\n",
            "${text}",
            "END OF POST.\n\n"
            "Perform aspect-based sentiment analysis (positive, neutral, negative) on the above post.\n\n",
            "If applicable to the post, choose aspects from the following:",
            f"{', '.join(aspects.FINANCIAL_ASPECTS)}\n\n",
            "If not applicable, you may choose other aspects that are more appropriate.\n\n",
            "Also include more personal aspects that may be helpful for potential investors, such as the performance of a particular company.\n\n",
            "Return a nested JSON object with the outer key being the index of the review, and the value being another JSON object with the aspect as the key and the sentiment as the value:\n",
            "The output should look like this:\n",
            _ABSA_OUTPUT_FORMAT,
        ]
    )
)

_ANNOTATED_ABSA_OUTPUT_FORMAT = """{'text': ['The ', ['CGI was amazing', 'Visual Effects', '#afa'], ' but I think that the ', ['acting was terrible', 'Acting', '#faa'], ' and I ', [' do not have an opinion on the plot', 'Plot', '#bebebe'], '.']}"""
ANNOTATED_ABSA_PROMPT = Template(
    "\n".join(
        [
            "The following is a text:\n",
            "${text}",
            "Perform aspect-based sentiment analysis (positive, neutral, negative) on the above text.\n\n",
            "Use the following aspects if they apply to the text: ",
            "${aspects}\n\n",
            "The output should be a JSON object. The outermost key should be 'text'.\n",
            "The value being a list of strings and lists of other strings: if a chunk of text has no annotation, leave it as be.\n",
            "If a chunk of text has an annotation, replace it with another list of the chunk of text, the aspect, and the color.\n",
            "Positive aspects should be '#afa', negative aspects should be '#faa', and neutral aspects should be '#bebebe'.\n\n"
            "For example, if the review is:\n",
            "The CGI was amazing but I think that the acting was terrible and I do not have an opinion on the plot",
            "The output should be:\n",
            _ANNOTATED_ABSA_OUTPUT_FORMAT,
        ]
    )
)

# output_format = '{"text": ["The ", ["CGI was amazing", "Visual Effects", "#afa"], " but I think that the ", ["acting was terrible", "Acting", "#faa"], " and I ", [" do not have an opinion on the plot", "Plot", "#bebebe"], "."]}'
