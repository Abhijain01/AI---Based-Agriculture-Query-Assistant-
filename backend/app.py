from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os, random, requests
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True) # allow frontend (React) to connect

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["mywebapp_db"]
users = db["users"]
otps = db["otps"]
listings_col = db["buyer_requests"]

# ✅ Home route
@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running 🚀"})

# ✅ Send OTP
@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.json
    phone = data.get("phone")
    if not phone:
        return jsonify({"error": "Phone number required"}), 400

    otp = str(random.randint(100000, 999999))
    otps.update_one({"phone": phone}, {"$set": {"otp": otp}}, upsert=True)
    return jsonify({"message": "OTP sent successfully", "otp": otp})

# ✅ Verify OTP
@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    phone = data.get("phone")
    otp = data.get("otp")

    record = otps.find_one({"phone": phone})
    if record and record["otp"] == otp:
        user = users.find_one({"phone": phone}, {"_id": 0})
        return jsonify({
            "message": "OTP verified ✅",
            "existing_user": bool(user),
            "user": user
        })
    return jsonify({"error": "Invalid OTP ❌"}), 400

# ✅ Register user
@app.route("/register-details", methods=["POST"])
def register_details():
    data = request.json
    phone = data.get("phone")

    if not phone:
        return jsonify({"error": "Phone required"}), 400

    crops = []
    for c in data.get("crops", []):
        crop = {
            "name": c.get("name"),
            "soil": c.get("soil"),
            "landArea": c.get("landArea"),
            "farmLocation": c.get("farmLocation"),
        }
        crops.append(crop)

    users.update_one(
        {"phone": phone},
        {
            "$set": {
                "name": data.get("name"),
                "email": data.get("email"),
                "state": data.get("state"),
                "address": data.get("address"),
                "district": data.get("district"),
                "pin": data.get("pin"),
                "crops": crops,
            }
        },
        upsert=True,
    )
    return jsonify({"message": "User details saved successfully"}), 200

# ✅ Get user by phone (for topbar)
@app.route("/get-user", methods=["GET"])
def get_user():
    phone = request.args.get("phone")
    if not phone:
        return jsonify({"error": "Phone required"}), 400

    user = users.find_one({"phone": phone}, {"_id": 0})
    if not user:
        return jsonify({"name": "Guest"})
    return jsonify(user)

# ✅ Save/update location for user
@app.route("/update-location", methods=["POST"])
def update_location():
    data = request.json
    phone = data.get("phone")
    lat = data.get("lat")
    lon = data.get("lon")
    city = data.get("city")

    if not phone or lat is None or lon is None or not city:
        return jsonify({"error": "phone, lat, lon, city required"}), 400

    users.update_one(
        {"phone": phone},
        {"$set": {"current_location": {"lat": lat, "lon": lon, "city": city}}},
        upsert=True,
    )

    return jsonify({"message": "Location updated ✅", "city": city})

# ✅ Weather using WeatherAPI.com
WEATHER_API = os.getenv("WEATHER_API", "6b8f15b41c80499c96c43953252409")

@app.route("/weather", methods=["GET"])
def weather():
    phone = request.args.get("phone")
    user = users.find_one({"phone": phone}, {"_id": 0})

    if user and "current_location" in user:
        lat = user["current_location"]["lat"]
        lon = user["current_location"]["lon"]
        try:
            url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API}&q={lat},{lon}"
            res = requests.get(url)
            data = res.json()
            return jsonify({
                "temp": f"{data['current']['temp_c']}°C",
                "humidity": f"{data['current']['humidity']}%",
                "wind": f"{data['current']['wind_kph']} km/h",
                "desc": data['current']['condition']['text'],
                "city": data['location']['name']
            })
        except Exception as e:
            print("Weather API error:", e)

    return jsonify({
        "temp": "28°C",
        "humidity": "65%",
        "wind": "10 km/h",
        "desc": "Partly Cloudy"
    })

# ✅ DUMMY total revenue API
@app.route("/market-revenue", methods=["GET"])
def market_revenue():
    return jsonify({"total": "₹1,24,500"})

# ✅ Buyer Requests API — fetch listings
@app.route("/buyer-requests", methods=["GET"])
def buyer_requests():
    if listings_col.count_documents({}) == 0:
        dummy_data = [
            {
                "id": "req1",
                "crop": "Maize",
                "buyer": "Asha Agri Traders",
                "qty": 500,
                "unit": "kg",
                "price": 2200,
                "image": "https://source.unsplash.com/480x320/?maize,corn",
                "location": "Ludhiana, PB",
                "postedAt": datetime.now().isoformat()
            },
            {
                "id": "req2",
                "crop": "Wheat",
                "buyer": "Punjab Buyers Co.",
                "qty": 1000,
                "unit": "kg",
                "price": 2100,
                "image": "https://source.unsplash.com/480x320/?wheat,field",
                "location": "Amritsar, PB",
                "postedAt": datetime.now().isoformat()
            },
            {
                "id": "req3",
                "crop": "Rice (Basmati)",
                "buyer": "Rao Traders",
                "qty": 200,
                "unit": "kg",
                "price": 3000,
                "image": "https://source.unsplash.com/480x320/?rice,field",
                "location": "Fazilka, PB",
                "postedAt": datetime.now().isoformat()
            },
            {
                "id": "req4",
                "crop": "Sugarcane",
                "buyer": "Harvest Hub",
                "qty": 50,
                "unit": "tons",
                "price": 18000,
                "image": "https://source.unsplash.com/480x320/?sugarcane,field",
                "location": "Ettumanoor, KL",
                "postedAt": datetime.now().isoformat()
            },
        ]
        listings_col.insert_many(dummy_data)

    listings = list(listings_col.find({}, {"_id": 0}))
    return jsonify({"requests": listings})

# ✅ Post a new buyer listing
@app.route("/post-listing", methods=["POST"])
def post_listing():
    data = request.json
    data["id"] = f"req{random.randint(1000,9999)}"
    data["postedAt"] = datetime.now().isoformat()
    listings_col.insert_one(data)
    return jsonify({"message": "Listing created successfully", "listing": data})

# ✅ Mark interest in listing
@app.route("/interest", methods=["POST"])
def mark_interest():
    listing_id = request.json.get("listingId")
    phone = request.json.get("phone")
    if not listing_id or not phone:
        return jsonify({"error": "listingId and phone required"}), 400

    listings_col.update_one({"id": listing_id}, {"$addToSet": {"interested": phone}})
    return jsonify({"message": "Interest recorded ✅"})

# ✅ Other small routes
@app.route("/news", methods=["GET"])
def news():
    return jsonify({"headline": "Monsoon expected to arrive early this year."})

@app.route("/schemes", methods=["GET"])
def schemes():
    return jsonify({"schemes": [
        {"title": "PM-KISAN", "desc": "Income support for farmers"},
        {"title": "Soil Health Card", "desc": "Check soil quality"}
    ]})

@app.route("/fertilizers", methods=["GET"])
def fertilizers():
    return jsonify({"items": [
        {"name": "Urea", "price": "₹300/bag"},
        {"name": "DAP", "price": "₹1200/bag"}
    ]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
