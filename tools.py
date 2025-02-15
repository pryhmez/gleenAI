
import requests

tools_info = {
    "MeetingScheduler": {
        "name": "MeetingScheduler",
        "description": "Books meetings with clients."
    },
    "GymAppointmentAvailability": {
        "name": "GymAppointmentAvailability",
        "description": "Assistants must Check next appointment available in Gym and confirm with custome, before offering appointment to customer"
    },
    "OnsiteAppointment": {
        "name": "OnsiteAppointment",
        "description": "Book Onsite Gym Appointment as per user's availability."
    },
    "PriceInquiry": {
        "name": "PriceInquiry",
        "description": "Fetches product prices of Gym memberships.",
        "parameters": {
            "product_name": ["Silver-Gym-Membership", "Gold-Gym-Membership", "Platinum-Gym-Membership"]
        }
    }
}

def onsite_appointment():
    # Assume arguments is a dict that contains date and time
    print('Onsite Appointment function is called')
    return f"Onsite Appointment is booked."

# def fetch_product_price(arguments):
#     print('Fetch product Price is called')
#     return f"The price is $29 per month."


def fetch_product_price(membership_type):
    print('Fetch product price is called')

    # Set up the endpoint and headers
    url = 'https://kno2getherworkflow.ddns.net/webhook/fetchMemberShip'
    headers = {'Content-Type': 'application/json'}
    
    # Prepare the data payload with the membership type
    data = {
        "membership": membership_type
    }
    
    # Send a POST request to the server
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response to get the price
        price_info = response.json()
        return f"The price is ${price_info['price']} per month."
    else:
        return "Failed to fetch the price, please try again later."

def calendly_meeting():
    print('Calendly Meeting invite is sent.')
    # Assume arguments is a dict that contains date and time
    return f"Calendly meeting invite is sent now."

def appointment_availability():
    print('Checking appointment availability.')
    # Assume arguments is a dict that contains date and time
    return f"Our next available appointment is tomorrow, 24th April at 4 PM."