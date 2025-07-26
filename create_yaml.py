render_yaml_content = """
services:
  - type: web
    name: my-streamlit-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run apprstr.py
    pythonVersion: 3.10.13
"""

with open("render.yaml", "w") as f:
    f.write(render_yaml_content.strip())

print("✅ render.yaml created successfully.")
