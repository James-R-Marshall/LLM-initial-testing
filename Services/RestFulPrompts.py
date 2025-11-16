from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
users = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

@app.route('/api/users', methods=['POST'])
def add_user():
    new_user = request.json
    users.append(new_user)
    return jsonify(new_user), 201

@app.route('/generate-resp', methods=['POST'])
def receive_large_text():
    if request.is_json:
        data = request.get_json()
        large_text = data.get('content')
        # Process the large_text here
        print(f"Received large text: {large_text[:100]}...") # Print first 100 chars
        save_path = "received_large_text.txt"
        with open(save_path, "w") as f:
            f.write(large_text)
        print(f"Large text saved to {save_path}")
        return jsonify({"message": "Data received successfully"}), 200
    return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run()