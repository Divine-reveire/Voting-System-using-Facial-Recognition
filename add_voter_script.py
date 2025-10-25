import requests

def add_voter():
    print("=== Add Voter Script ===")
    print("Enter voter details:")

    name = input("Name: ").strip()
    dob = input("Date of Birth (YYYY-MM-DD): ").strip()
    aadhar = input("Aadhar Number (12 digits): ").strip()
    voterid = input("Voter ID (10 alphanumeric): ").strip()
    phone = input("Phone Number (10 digits): ").strip()

    # Basic validation
    if not all([name, dob, aadhar, voterid, phone]):
        print("Error: All fields are required.")
        return

    if len(aadhar) != 12 or not aadhar.isdigit():
        print("Error: Aadhar must be exactly 12 digits.")
        return

    if len(voterid) != 10 or not voterid.isalnum():
        print("Error: Voter ID must be exactly 10 alphanumeric characters.")
        return

    if len(phone) != 10 or not phone.isdigit():
        print("Error: Phone must be exactly 10 digits.")
        return

    # Send to API
    try:
        response = requests.post('http://127.0.0.1:5000/add_voter', json={
            'name': name,
            'dob': dob,
            'aadhar': aadhar,
            'voterid': voterid,
            'phone': phone
        })

        data = response.json()
        if data.get('success'):
            print("✅ Voter added successfully!")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error connecting to server: {e}")

if __name__ == "__main__":
    while True:
        add_voter()
        again = input("Add another voter? (y/n): ").strip().lower()
        if again != 'y':
            break
    print("Done.")
