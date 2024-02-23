import re
from datasets import load_dataset
import pandas as pd


def transform_question(question):
    pattern = (
        r"TITLE: (.+) QUESTION \[(\d+) upvotes\]: (.+) REPLY \[(\d+) votes\]: (.+)"
    )

    match = re.match(pattern, question)

    if match:
        title = match.group(1)
        question = match.group(2)
        reply = match.group(3)

        output_string = f"Conversation between a Human and a proofer assistant [INS] {title}: {question} [/INS] {reply}"

        return output_string
    else:
        return "Input format not recognized."


def transform_paper(paper):
    pattern = r"\\begin{document} \\title\{(.+?)\} (.+?)\\bibliographystyle(.+)"

    match = re.match(pattern, paper)

    if match:
        title = match.group(1)
        content = match.group(2)

        output_string = f"Conversation between a Human and a proofer assistant  [INS] {title} [/INS] {content}"

        return output_string
    else:
        return "Input format not recognized."


def transform_sample(sample):
    try:
        text = sample.get("text")
        meta = sample.get("meta")

        if "question_id" in meta:
            new_text = transform_question(text)
            new_sample = {"text": new_text, "meta": meta}
            return new_sample
        elif "config" in meta:
            if text.startswith("\begin"):
                new_text = transform_paper(text)
                new_sample = {"text": new_text, "meta": meta}
                return new_sample

        return sample

    except Exception as e:
        print("transform_sample error " + e)
        return sample


def main():

    #     x = """TITLE: Evaluating the limit $\lim_{n\to+\infty}(\sqrt[n]{n}-1)^n$ QUESTION [6 upvotes]: Evaluate the limit $$\lim_{n\to+\infty}(\sqrt[n]{n}-1)^n$$ I know the limit is 0 by looking at the graph of the function, but how can I algebraically show that that is the limit? REPLY [1 votes]: Since, $\frac{\log(x)}x\le\frac1e$, we have that $$ \begin{align} \sqrt[n]{n}-1 &\le e^{1/e}-1\\ &\lt1 \end{align} $$ Therefore, $$ \begin{align} \lim_{n\to\infty}\left(\sqrt[n]{n}-1\right)^n &\le\lim_{n\to\infty}\left(e^{1/e}-1\right)^n\\[3pt] &=0 \end{align} $$
    # """
    #     y = transform_sample({"text": str(x), "meta": {"question_id": 123}})
    #     print(y)

    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="train",
        trust_remote_code=True,
    )

    df = pd.DataFrame(dataset)
    df.to_csv("dataset-train.csv", index=False)

    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="test",
        trust_remote_code=True,
    )

    df = pd.DataFrame(dataset)
    df.to_csv("dataset-test.csv", index=False)

    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="validation",
        trust_remote_code=True,
    )

    df = pd.DataFrame(dataset)
    df.to_csv("dataset-validation.csv", index=False)


if __name__ == "__main__":
    main()
