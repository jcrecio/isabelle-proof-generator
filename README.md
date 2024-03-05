# Instructions to setup isabelle-proof-generator

1. Install poetry in the computer
2. Create a virtual environment:
    ```{shell}
    poetry shell  
    ```
3. Install the dependencies:
    ```{shell}
    poetry install  
    ```
3. Raw Math dataset: proof-pile

    https://huggingface.co/datasets/hoskinson-center/proof-pile

4. Get the raw dataset and map it to our format for Mistral:
    ```{shell}
    <INTRODUCTION>[INS]<QUESTION>[/INS]<ANSWER>
    ```
    Run the command:
    ```{shell}
    poetry run python .\baldurcito\prepare_dataset.py   
    ```

