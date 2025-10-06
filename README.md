# Aggie XLMA
A Python-based application that helps you view, analyze and interact with LMA ([Lightning Mapping Array](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004JD004549)) data. This application replicates [XLMA's](http://www.lightning.nmt.edu/nmt_lms/steps_2000/more_rt.html) essential features while operating significantly faster. It is not meant as an alternative and does not offer full functionality. We suggest checking out the [PyXLMA](https://github.com/deeplycloudy/xlma-python) project for that purpose. Our mission is to perform the most used tasks of viewing, selection, and extracting flash information reliably and quickly with a modern, responsive UI.

## Installation instructions:

1. Clone the repository using: `git clone https://github.com/krishnacalindi/hlma.git`.
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual enviroment. This can vary depending on your [OS](https://python.land/virtual-environments/virtualenv#Python_venv_activation).
4. Install required libraries: `pip install -r requirements.txt`
5. Run the application using `python hlma.py`.

## Features:

1. GUI based: we offer a GUI heavily influenced by feedback from LMA scientists.
   ![GUI](assets/images/example.png)
2. Filtering: We offer the ability to filter the data using specified filter fields, or selecting with a polygon.
3. Plotting: We provide a suite of map features, map options and [colorcet's](https://colorcet.holoviz.org/user_guide/Continuous.html#linear-sequential-colormaps-for-plotting-magnitudes) entire collection of named linear colormaps.
4. Exporting: You can export LMA data as .dat files that are compatible with XLMA and lmatools and parquet files for further processing.
5. States: You can save the state of the application at any point, and use the  saved file to get back to where you were!
6. Flash algorithms: We offer simple dot-to-dot and McCaul flash algorithm via a custom implementation. This feature is still in beta, please be cautious of errors.

## Contact:

If you would like to contribute please reach out to [Dr.Timothy Logan](https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html).
