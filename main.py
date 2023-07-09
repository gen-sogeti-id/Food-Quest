from flask import Flask, render_template, request

#import vertexai
#from vertexai.preview.language_models import TextGenerationModel
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

def food_quest(content):
#    vertexai.init(project="528941013002", location="us-central1")
#    parameters = {
#        "temperature": 0.2,
#        "max_output_tokens": 256,
#        "top_p": 0.8,
#        "top_k": 40
#    }
#    model_name = "projects/528941013002/locations/us-central1/models/1648134944687390720"
#    model = TextGenerationModel.from_pretrained("text-bison@001")
#    model = model.get_tuned_model(model_name)
#    response = model.predict(
#        content,
#        **parameters
#    )

    vertexai.init(project="gen-sogeti-id", location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 880,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
        context="""Objective/Goal
    •	Provide a personalized recommendation based on prompts
    •	Personal assistant for Food choices
    •	Category of Food – Spicy, Non Spicy – Levels
    •	Carbs, Protein, Fiber
    •	Menus of the restaurants
    •	Health information – chronic disease, Fitness goals, genetics (allergies, diabetes, heart condition)
    •	Build LLMs using restaurants, nutrition association books, medical association books
    •	Enhance and make it easy for the nutritionist as well as the customer (someone trying to eat healthy, conscious eaters).
    """,
        examples=[
            InputOutputTextPair(
                input_text="""What are some Gluten Free restaurant options in Denver Colorado?""",
                output_text="""Acova at 3651 Navajo Street, Denver, CO 80211. 
    When the Workmans opened Acova, they decided to make it as safe as possible for those with Celiac disease.
    The Hornet at 76 Broadway, Denver, CO 80203. 
    The Hornet, which harnesses MyMeal to keep guests informed of safety practices and has a wide variety of gluten-free specialties.
    Quiero Arepas at 3200 N Pecos St, Denver, CO 80211.
    Quiero Arepas® is a 100% gluten free establishment, our entire menu is all natural and features the Venezuelan arepa. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest some Italian food places with location in Chicago, Illinois and address ?""",
                output_text="""• Il Porcellino at 205 W Kinzie St, Chicago, IL 60610. Il Porcellino is a great place to get Italian food in Chicago. They have a wide variety of pasta dishes, as well as pizza, sandwiches, and salads. The service is excellent and the atmosphere is cozy and inviting. 
                • The Purple Pig at 500 N Clark St, Chicago, IL 60654. The Purple Pig is another great option for Italian food in Chicago. They have a wide variety of small plates, as well as pasta dishes, pizza, and sandwiches. The service is excellent and the atmosphere is lively and fun. 
                • The Girl & the Goat at 800 W Randolph St, Chicago, IL 60607. The Girl & the Goat is a Michelin-starred restaurant that serves modern American cuisine with Italian influences. They have a wide variety of dishes, including pasta, pizza, and sandwiches. The service is excellent and the atmosphere is elegant and sophisticated. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest some Indian food places with location in Chicago, Illinois and address ?""",
                output_text="""Sure, here are some Indian food places with location in Chicago, Illinois and address: 
                * The Rice Boat, 1800 N Clark St, Chicago, IL 60614 
                * The Spice Room, 1038 W Randolph St, Chicago, IL 60607 
                * The Punjab Palace, 1034 W Devon Ave, Chicago, IL 60660 
                * The India House, 201 E Jackson Blvd, Chicago, IL 60604 
                * The Bombay Palace, 1725 N Clark St, Chicago, IL 60614. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest some Indian food places with location in Denver, CO and address ?""",
                output_text="""Sure, here are some Indian food places in Denver, CO: 
                1. The Curry Leaf, 1400 E Colfax Ave, Denver, CO 80206 
                2. India Palace, 1001 E Colfax Ave, Denver, CO 80206 
                3. India Bistro, 1215 E Colfax Ave, Denver, CO 80206 
                4. India Grill, 1500 E Colfax Ave, Denver, CO 80206 
                5. India Oven, 1500 E Colfax Ave, Denver, CO 80206. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest some American food places with location in Dallas, Texas and address ?""",
                output_text="""Sure, here are some American food places with location in Dallas, Texas and address: 
                • The Rustic - 3010 Knox St, Dallas, TX 75205 
                • The Cheesecake Factory - 600 North Park Center, Dallas, TX 75225 
                • Pappadeaux Seafood Kitchen - 4400 McKinney Ave, Dallas, TX 75205 
                • The Capital Grille - 1700 McKinney Ave, Dallas, TX 75201 
                • Del Frisco's Double Eagle Steakhouse - 2500 McKinney Ave, Dallas, TX 75201. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest some healthy food places with location in Dallas, Texas and address ?""",
                output_text="""Sure, here are some healthy food places with location in Dallas, Texas with address: 
                1. The Good Seed Cafe, 3015 Oak Lawn Ave, Dallas, TX 75219 
                2. True Food Kitchen, 2300 McKinney Ave, Ste 100, Dallas, TX 75201 
                3. Flower Child, 4400 Belt Line Rd, Ste 100, Dallas, TX 75252 
                4. The Henry, 1722 N Lamar St, Dallas, TX 75202 5. Snap Kitchen, 4210 McKinney Ave, Ste 100, Dallas, TX 75204. """
            ),
            InputOutputTextPair(
                input_text="""Can you please give diet chart for loosing 10 lbs in a week ? I am diabetic and vegeterian.""",
                output_text="""A diet chart for losing 10 lbs in a week is not recommended for anyone, especially for diabetics. 
                It is important to lose weight slowly and steadily in order to maintain a healthy weight loss. 
                A sudden and drastic weight loss can lead to health problems such as fatigue, dizziness, and hair loss. 
                It can also make it difficult to stick to a healthy diet and exercise plan in the long term. 
                If you are diabetic and want to lose weight, it is important to talk to your doctor first. 
                They can help you create a diet and exercise plan that is safe and effective for you. """
            ),
            InputOutputTextPair(
                input_text="""Can you please give diet chart for loosing 10 lbs in a month ? I am diabetic and vegeterian.""",
                output_text="""Sure, here is a sample diet chart for losing 10 lbs in a month for a diabetic vegetarian: 
                * Breakfast: Oatmeal with fruit and nuts 
                * Lunch: Salad with grilled chicken or tofu 
                * Dinner: Vegetable stir-fry with brown rice 
                * Snacks: Yogurt with fruit or nuts, whole-wheat crackers with hummus, fruit, or vegetables. 
                Be sure to talk to your doctor before starting any new diet, and make sure to make gradual changes to your eating habits so that you can stick to them long-term. """
            ),
            InputOutputTextPair(
                input_text="""Can you please suggest calories based menu for loosing 30 lbs in next three months ? I am heart patient and non-vegeterian.""",
                output_text="""Sure, I can help you with that. 
                Here is a sample menu for you to follow: 
                * Breakfast: Oatmeal with fruit and nuts 
                * Lunch: Salad with grilled chicken or fish 
                * Dinner: Lean protein with vegetables 
                * Snacks: Yogurt with fruit or nuts This menu is designed to help you lose weight in a healthy way. 
                It is low in calories and fat, and it is also high in fiber and protein. 
                This will help you feel full and satisfied, and it will also help you burn fat. 
                In addition to following this menu, you should also exercise regularly. 
                This will help you burn calories and lose weight even faster. 
                Aim for at least 30 minutes of moderate-intensity exercise most days of the week. 
                If you have any other questions or concerns, please do not hesitate to ask. 
                I am here to help you reach your weight loss goals. """
            ),
            InputOutputTextPair(
                input_text="""Could you please suggest calories based menu for American restaurants in Dallas, Texas ?""",
                output_text="""Sure, here are some options: 
                • The Cheesecake Factory: This restaurant has a wide variety of American dishes, including burgers, pizzas, and pasta. Most of their dishes are under 500 calories. 
                • Chili's: This restaurant is known for its burgers and Tex-Mex dishes. Most of their dishes are under 600 calories. 
                • Applebee's: This restaurant has a variety of American dishes, including salads, burgers, and pasta. Most of their dishes are under 700 calories. 
                • Olive Garden: This restaurant is known for its Italian dishes, including pasta, pizza, and salads. Most of their dishes are under 800 calories. """
            ),
            InputOutputTextPair(
                input_text="""Could you please suggest up to 500 calories based menu for American restaurants in Denver, CO?""",
                output_text="""Sure, here are some options: 
                - The Original Pancake House: The Original Pancake House has a variety of American breakfast options that are all under 500 calories. 
                For example, you can get a plain pancake for 160 calories, or a fruit plate for 150 calories. 
                - The Pantry: The Pantry is a classic American diner that also has a number of healthy options under 500 calories. 
                For example, you can get a grilled cheese sandwich for 270 calories, or a turkey burger with lettuce and tomato for 350 calories. 
                - The Cheesecake Factory: The Cheesecake Factory is a popular American restaurant that also has a number of healthy options under 500 calories. 
                For example, you can get a grilled chicken salad for 350 calories, or a roasted vegetable pizza for 400 calories. """
            )
        ]
    )

    response = chat.send_message(content, **parameters)
    return response

app = Flask(__name__,)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    input_text = request.args.get("input")
    recommendation = food_quest(input_text)
    return render_template('process.html', recommendation=recommendation)

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)