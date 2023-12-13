from string import Template


MOVIE_SUMMARY_PROMPT = Template(
    "\n".join(
        [
            "The following is a list of movie reviews:\n",
            "${text}",
            "END OF REVIEWS.\n\n",
            "Summarize the above reviews with the title 'Summary'\n\n",
            "Next, in a new paragraph, with the title 'Watch this if...', describe who would like this movie.\n\n",
            "In another paragraph, with the title 'Skip this if..., and who would not like this movie and why.\n\n",
            "Finally, in a section titled 'Recommendations', recommend 10 other movies that are similar to this one. Include the title of the movie, the directory, and the year it was released.\n\n",
            "Use markdown to format the output. Use any sort of formatting to be more appealing to the audience.\n\n",
        ]
    )
)

FINANCE_SUMMARY_PROMPT = Template(
    "\n".join([
        "以下是一篇財經文章：\n",
        "${text}",
        "END OF POST.\n\n",
        "摘要上述文章，並將摘要標題設為 '摘要'。\n\n",
        "接著，以 '這對誰有利' 為標題，判斷目前股市情況對何種投資者是有利的。\n\n",
        "再來，以 '這對誰不利' 為標題，判斷目前股市情況對何種投資者是不利的。\n\n",
        "使用 MARKDOWN 來格式化輸出。使用任何格式來使輸出更吸引人。\n\n",
    ])
)