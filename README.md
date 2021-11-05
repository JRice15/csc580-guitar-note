# Guitar Note Tabulator

## Conda env

Create the conda environment as follows:
`conda env create -f tf2.4-env.yml -n <name>`

where `<name>` is what name you want to call this new environment (I called mine "guitartab").


## Training

`python3 train.py --model <model-name> --name <optional-name-to-save-under>`

## Using a model

On audio from a file:

`python3 generate_tabulature.py --name <optional-name> --file <filename> --saveas <optional-output-tab-filename>`

On live audio (from builtin microphone):

`python3 generate_tabulature.py --name <optional-name> --live --saveas <optional-output-tab-filename>`

