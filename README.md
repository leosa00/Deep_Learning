Setup and Installation

Follow these steps to create the necessary Python environment and install all dependencies for the project.

1. Create a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

python3 -m venv venv


2. Activate the Environment

You must activate the environment before running any code.

On macOS/Linux:

source venv/bin/activate


On Windows (Command Prompt):

venv\Scripts\activate.bat


On Windows (PowerShell):

venv\Scripts\Activate.ps1


3. Install Required Packages

Once the environment is active, install the project dependencies listed in requirements.txt.

pip install -r requirements.txt
