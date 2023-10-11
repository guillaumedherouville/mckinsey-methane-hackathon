# Methane-hackathon
This is a copy of the README

A methane detection project using satellite images

##  Requirements

### System Requirements
- Python 3.9 or higher

### Packages
```setup
pip install -r requirements.txt
```

## Structure
The structure of the repository is as follows : 

````
methane-hackathon
┣ app/ --> hosts the streamlit webapp
┣ data/ --> for all data files during development
┃ ┣ interim/
┃ ┣ processed/
┃ ┗ raw/
┣ docs/ --> includes docs file (where index.md is a copy of the readme)
┣ logs/ --> stores models & results 
┣ models/ --> store models 
┣ notebooks/ --> for exploration and trial
┣ output/ --> submission file
┣ tests/ --> for smooth CI/CD
┣ utils/ --> useful misc scripts
┣ .gitignore
┣ mkdocs.yml
┣ README.md
┗ requirements.txt 
````

## Running the project
To navigate the project, we suggest :

- View the documentation with 'mkdocs-serve' (to be hosted on github pages when the repo will be made public)
  
- Run the app on https:// ...   (for more details, see the documentation)

## To-do list:
- Create tests
- Finish the app
- Add docstrings
