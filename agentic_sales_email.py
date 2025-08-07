# pip install sendgrid openai-agents
from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from typing import Dict
import sendgrid
import os
import asyncio
from sendgrid.helpers.mail import Mail, Email, To, Content

load_dotenv(override=True)

# --------------------- configuration variables ---------------------
llm_model = "gpt-4o-mini"
sender_email = os.environ["SENDER_EMAIL"]
receiver_email = os.environ["RECEIVER_EMAIL"]

# --------------------- instruction prompts -------------------------
prof_instructions = "You are a sales agent working for idare.ai, \
a company that provides zero-code predictive analytics solution powered by AI. \
You write professional, serious cold emails."

witty_instructions = "You are a humorous, engaging sales agent working for idare.ai, \
a company that provides zero-code predictive analytics solution, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."

busy_instructions = "You are a busy sales agent working for idare.ai, \
a company that provides provides zero-code predictive analytics solution, powered by AI. \
You write concise, to the point cold emails."

subject_instructions = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

sales_manager_instructions = """
You are a Sales Manager at idare.ai. Your goal is to find the single best cold sales email using the sales tools.

Follow these steps carefully:
1. Generate Drafts: Use all three sales tools (prof_sales, witty_sales, busy_sales) to generate three different email 
drafts. Do not proceed until all three drafts are ready.

2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most 
effective.

3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take 
care of formatting and sending.

Crucial Rules:
- You must use the sales agent tools to generate the drafts — do not write them yourself.
- You must hand off exactly ONE email to the Email Manager — never more than one.
"""

email_manager_instructions =("You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert "
                             "the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body.")

prompt = ("Send out a cold sales email addressed to Dear CTO Dr. Khairul Chowdhury from Kawsar, Data Scientist "
          "at idare.ai")

# --------------------- Create Agents -------------------------
prof_sales_agent = Agent(name="Professional Sales Agent",
                         instructions=prof_instructions,
                         model=llm_model)

witty_sales_agent = Agent(name="Engaging Sales Agent",
                          instructions=witty_instructions,
                          model=llm_model)

busy_sales_agent = Agent(name="Busy Sales Agent",
                         instructions=busy_instructions,
                         model=llm_model)

subject_writer = Agent(name="Email subject writer",
                       instructions=subject_instructions,
                       model=llm_model)

html_converter = Agent(name="HTML email body converter",
                       instructions=html_instructions,
                       model=llm_model)

# --------------------- Create Tools -------------------------
sales_agent_description = "Write a cold sales email"
prof_sales = prof_sales_agent.as_tool(tool_name="prof_sales_agent",
                                      tool_description=sales_agent_description)
witty_sales = witty_sales_agent.as_tool(tool_name="witty_sales_agent",
                                        tool_description=sales_agent_description)
busy_sales = busy_sales_agent.as_tool(tool_name="busy_sales_agent",
                                      tool_description=sales_agent_description)

subject_tool = subject_writer.as_tool(tool_name="subject_writer",
                                      tool_description="Write a subject for a cold sales email")

html_tool = html_converter.as_tool(tool_name="html_converter",
                                   tool_description="Convert a text email body to an HTML email body")

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    try:
        sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        from_email = Email(sender_email)  # Change to your verified sender
        to_email = To(receiver_email)  # Change to your recipient
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_email, subject, content).get()
        sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --------------------- Create Agents -------------------------
emailer_agent = Agent(name="Email Manager",
                      instructions=email_manager_instructions,
                      tools=[subject_tool, html_tool, send_html_email],
                      model=llm_model,
                      handoff_description="Convert an email to HTML and send it")

sales_manager = Agent(name="Sales Manager",
                      instructions=sales_manager_instructions,
                      tools=[prof_sales, witty_sales, busy_sales],
                      handoffs=[emailer_agent],
                      model=llm_model)

# --------------------- main ----------------------------------
async def main(query: str):
    with trace("Automated SDR"):
        result = await Runner.run(sales_manager, query)
        print(result)

if __name__ == "__main__":
    asyncio.run(main(prompt))
