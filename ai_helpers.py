import openai
from config import Config
import json
from langchain_core.prompts import PromptTemplate
from prompts import AGENT_STARTING_PROMPT_TEMPLATE, STAGE_TOOL_ANALYZER_PROMPT, AGENT_PROMPT_OUTBOUND_TEMPLATE, AGENT_PROMPT_INBOUND_TEMPLATE
from flask import session  # Uncomment it after testing.
from stages import OUTBOUND_CONVERSATION_STAGES, INBOUND_CONVERSATION_STAGES
from tools import tools_info,onsite_appointment,fetch_product_price,calendly_meeting,appointment_availability
from groq import Groq

# session ={} # Added for testing. remove after testing

openai.api_key = Config.OPENAI_API_KEY
salesperson_name = Config.AISALESAGENT_NAME
company_name = Config.COMPANY_NAME
company_business = Config.COMPANY_BUSINESS
conversation_purpose = Config.CONVERSATION_PURPOSE
company_products_services = Config.COMPANY_PRODUCT_SERVICES
conversation_stages = OUTBOUND_CONVERSATION_STAGES
gclient = Groq(api_key='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')

def sanitize_json_string(json_string):
    """Sanitize JSON string to remove unnecessary escape characters."""
    return json_string.replace("\\", "")

def gen_ai_output(prompt):
    response = gclient.chat.completions.create(
                # model="mixtral-8x7b-32768",
                model="llama-3.3-70b-versatile",
                messages=prompt,
                temperature=0.5,
                max_tokens=100,
                stream=False,
                top_p=1
            )    
    # response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=prompt,
    #             temperature=0.5,
    #             max_tokens=100,
    #         )
    return response.choices[0].message.content

def is_tool_required(ai_output):
    """Check if the use of a tool is required according to AI's output."""
    try:
        data = json.loads(ai_output)
        return data.get("tool_required") == "yes"
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in AI output.")

def get_conversation_stage(ai_output):
    """Extract the conversation stage from AI's output."""
    try:
        ai_output = sanitize_json_string(ai_output)
        data = json.loads(ai_output)
        return int(data.get("conversation_stage_id"))
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in AI output.")

def get_tool_details(ai_output):
    """Retrieve the tool name and parameters if a tool is required."""
    ai_output = sanitize_json_string(ai_output)
    if not is_tool_required(ai_output):
        return None, None
    try:
        data = json.loads(ai_output)
        tool_name = data.get("tool_name")
        tool_parameters = data.get("tool_parameters")
        return tool_name, tool_parameters
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in AI output.")

def process_initial_message(customer_name, customer_problem):
    
    initial_prompt = AGENT_STARTING_PROMPT_TEMPLATE.format(
        salesperson_name=salesperson_name,
        company_name=company_name,
        company_business=company_business,
        conversation_purpose=conversation_purpose,
        conversation_stages=conversation_stages
        )
    message_to_send_to_ai=[
        {
            "role": "system",
            "content": initial_prompt
        }
    ]
    initial_transcript = "Customer Name:" + customer_name + ". Customer filled up details in the website:" + customer_problem
    message_to_send_to_ai.append({"role": "user", "content": initial_transcript})
    
    response = gen_ai_output(message_to_send_to_ai)
    return response

def invoke_stage_tool_analysis(message_history,user_input):
    conversation_stage_id = session.get('conversation_stage_id', 1)
    tools_description = "\n".join([
    f"{tool['name']}: {tool['description']}" +
    (f" (Parameters: {', '.join([f'{k} - possible values: {v}' if isinstance(v, list) else f'{k} - format: {v}' for k, v in tool.get('parameters', {}).items()])})" if 'parameters' in tool else "")
    for tool in tools_info.values()
    ])
    
    
    intent_tool_prompt= STAGE_TOOL_ANALYZER_PROMPT.format(
        salesperson_name=salesperson_name,
        company_name=company_name,
        company_business=company_business,
        conversation_purpose=conversation_purpose,
        conversation_stage_id=conversation_stage_id,
        conversation_stages=conversation_stages,
        conversation_history=message_history,
        company_products_services=company_products_services,
        user_input=user_input,
        tools=tools_description
    )
    message_to_send_to_ai=[
        {
            "role": "system",
            "content": intent_tool_prompt
        }
    ]
    message_to_send_to_ai.append({"role": "user", "content": "You Must Respond in the json format specified in system prompt"})
    ai_output = gen_ai_output(message_to_send_to_ai)

       # Log the raw AI output
    print("Raw AI Output:", ai_output)
    return ai_output


def initiate_inbound_message():
    initial_response = AGENT_PROMPT_INBOUND_TEMPLATE.format(
        salesperson_name=salesperson_name,
        company_name=company_name
    )
    return initial_response
    


def process_message(message_history, user_input):
    # if 'message_history' not in session:
    #     session['message_history'] = []
    # print("AI Tool and Conversation stage is decided: ", ai_output)
    """Process the AI decision to either call a tool or handle conversation stages."""
    print(message_history)
    print('user said:' + user_input)
    stage_tool_output=invoke_stage_tool_analysis(message_history, user_input)
    # print("stage tool output----------------- " + stage_tool_output )
    stage = get_conversation_stage(stage_tool_output)
    session['conversation_stage_id'] = stage
    tool_output=''
    try:
        if is_tool_required(stage_tool_output):
            print('Tool Required is true')
            tool_name, params = get_tool_details(stage_tool_output)
            print('Tool called'+ tool_name + 'tool param is: ' + params)
            if tool_name == "MeetingScheduler":
                tool_output = calendly_meeting()  # Assuming no parameters needed
                message_history.append({"role": "api_response", "content": tool_output})
            elif tool_name == "OnsiteAppointment":
                tool_output = onsite_appointment()  # Assuming no parameters needed
                message_history.append({"role": "api_response", "content": tool_output})
            elif tool_name == "GymAppointmentAvailability":
                tool_output = appointment_availability()  # Assuming no parameters needed
                message_history.append({"role": "api_response", "content": tool_output})                
            elif tool_name == "PriceInquiry":
                # Ensure params is a dictionary and contains 'product_name'
                tool_output = fetch_product_price(params)
                message_history.append({"role": "api_response", "content": tool_output})
            else:
                return ""
    except ValueError as e:
        tool_output = "Some Error Occured In calling the tools. Ask User if its okay that you callback the user later with answer of the query."
    
    conversation_stage_id = session.get('conversation_stage_id', 1)
    print("Creating inbound prompt template")
    inbound_prompt = AGENT_PROMPT_OUTBOUND_TEMPLATE.format(
        salesperson_name=salesperson_name,
        company_name=company_name,
        company_business=company_business,
        conversation_purpose=conversation_purpose,
        conversation_stage_id=conversation_stage_id,
        company_products_services=company_products_services,
        conversation_stages=json.dumps(conversation_stages, indent=2),
        conversation_history=json.dumps(message_history, indent=2),
        tools_response=tool_output
    )
    message_to_send_to_ai_final=[
        {
            "role": "system",
            "content": inbound_prompt
        }
    ]
    message_to_send_to_ai_final.append({"role": "user", "content": user_input})
    # session['message_history'].append({"role": "user", "content": user_input})
    print("Calling With inbound template: ", json.dumps(message_history))
    talkback_response = gen_ai_output(message_to_send_to_ai_final)
    return talkback_response


# Test Section : Remove After Testing
#--------------------------------------


# Initial customer details
# customer_name = "John Doe"
# customer_problem = "Looking for a gym membership to address back pain"

# # Initialize session
# session['message_history'] = []
# session['conversation_stage_id'] = 1

# initial_transcript = "Customer Name:" + customer_name + ". Customer filled up details in the website:" + customer_problem
# session['message_history'].append({"role": "user", "content": initial_transcript})

# # Generate initial message
# initial_response = process_initial_message(customer_name, customer_problem)
# session['message_history'].append({"role": "assistant", "content": initial_response})

# # Display initial AI response
# print("Assistant Response:", initial_response)

# # # Interactive loop to simulate conversation
# # while session.get('conversation_stage_id', 1) < 8:
# #     user_input = input("Your response: ")
# #     assistant_response = process_message(session['message_history'], user_input)
# #     session['message_history'].append({"role": "assistant", "content": assistant_response})
# #     stage_tool_output=invoke_stage_tool_analysis(session['message_history'], 'Based on last assistant response determine conversation stage')
# #     stage = get_conversation_stage(stage_tool_output)
# #     session['conversation_stage_id'] = stage
# #     print("Assistant Response:", assistant_response)
    
# while True:
#     user_input = input("Your response: ")
    
#     # Generate the assistant's response based on user input and message history
#     assistant_response = process_message(session['message_history'], user_input)
    
#     # Append the assistant's response to the message history
#     session['message_history'].append({"role": "assistant", "content": assistant_response})
    
#     # Check if the current assistant response contains 'END_OF_CALL'
#     if "<END_OF_CALL>" in assistant_response:
#         print("Assistant Response:", assistant_response)
#         print("The conversation has ended.")
#         break    
#     # Print the assistant's response
#     print("Assistant Response:", assistant_response)