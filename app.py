from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
	return 'Hello, World!'

# Ensures the server is only run if script is executed directly,
# not imported as a module
if __name__ == '__main__':
	app.run()