# Methane-hackathon
A methane detection project using satellite images

-> View the documentation on https ://
  
-> Run the app on https:// 

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

- View the documentation on https:// ...
-> Alternatively, you can run 'mkdocs serve' in your terminal to have a local host
  
- Run the app on https:// ...   (for a more extensive description, see the documentation)
