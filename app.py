import streamlit as st
import pandas as pd
import os
from langchain_groq.chat_models import ChatGroq
import logging
import dateparser

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("Advanced Query Interface with ChatGroq LLM")

api_key = st.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API Key to proceed.")
else:
    try:
        llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            api_key=api_key
        )
        logging.info("ChatGroq LLM successfully initialized.")
        st.success("Groq API Key successfully validated. You can now upload an Excel file.")
        
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

        if uploaded_file is None:
            st.warning("Please upload an Excel file.")
        else:
            def preprocess_data(data):
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    if data['date'].dt.tz is not None:
                        data['date'] = data['date'].dt.tz_localize(None)
                data = data[['sender', 'receiver', 'date', 'subject', 'body']]
                return data

            def validate_excel(data):
                required_columns = ['sender', 'receiver', 'date', 'subject', 'body']
                for column in required_columns:
                    if column not in data.columns:
                        st.error(f"Missing expected column: {column}")
                        return False
                return True

            try:
                data = pd.read_excel(uploaded_file, parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, utc=True))
                st.write("DataFrame preview:")
                st.dataframe(data.head())

                if not validate_excel(data):
                    st.stop()

                data = preprocess_data(data)
                st.success("Excel file successfully uploaded and processed. You can now ask queries.")

                query = st.text_input("Enter your query:")

                if st.button("Submit Query"):
                    if query:
                        def get_emails_on_date(data, date_str):
                            try:
                                date = dateparser.parse(date_str).date()
                                emails_on_date = data[data['date'].dt.date == date]
                                return len(emails_on_date)
                            except:
                                return None

                        def generate_data_summary(data):
                            summary = f"This dataset contains {len(data)} emails with the following columns: {', '.join(data.columns)}.\n"
                            summary += f"Date range: from {data['date'].min()} to {data['date'].max()}.\n"
                            summary += f"Top 5 senders: {', '.join(data['sender'].value_counts().nlargest(5).index)}\n"
                            summary += f"Top 5 receivers: {', '.join(data['receiver'].value_counts().nlargest(5).index)}\n"
                            return summary

                        def handle_query(query, data):
                            try:
                                if "how many email" in query.lower() and "on" in query.lower():
                                    date_str = query.lower().split("on")[-1].strip()
                                    email_count = get_emails_on_date(data, date_str)
                                    if email_count is not None:
                                        st.write(f"Query result: You received {email_count} emails on {date_str}.")
                                    else:
                                        st.error("Unable to process the date in your query. Please try a different format.")
                                elif "which mail is from" in query.lower():
                                    company = query.lower().split("from")[-1].strip()
                                    matching_emails = data[data['sender'].str.contains(company, case=False, na=False)]
                                    
                                    if not matching_emails.empty:
                                        st.write(f"Emails from {company}:")
                                        for _, email in matching_emails.iterrows():
                                            st.write(f"Subject: {email['subject']}")
                                            st.write(f"Sender: {email['sender']}")
                                            st.write(f"Date: {email['date']}")
                                            st.write("---")
                                    else:
                                        st.write(f"No emails found from {company}.")
                                else:
                                    data_summary = generate_data_summary(data)
                                    
                                    context = f"You are analyzing an email dataset. Here's a summary of the data:\n{data_summary}\n"
                                    context += "Here's a sample of 5 random emails from the dataset:\n"
                                    
                                    for _, row in data.sample(n=5).iterrows():
                                        context += f"Sender: {row['sender']}, Receiver: {row['receiver']}, Date: {row['date']}, "
                                        context += f"Subject: {row['subject']}, Body: {row['body'][:100]}...\n\n"
                                    
                                    context += "Please answer the following query based on this dataset: "
                                    
                                    full_prompt = context + query
                                    
                                    st.info(f"Processing query: {query}")
                                    
                                    response = llm.invoke(full_prompt)
                                    
                                    answer = response.content
                                    
                                    st.write(f"Query result: {answer}")
                            except Exception as e:
                                st.error(f"Error processing query: {e}")
                                logging.error(f"Error processing query: {e}")

                        handle_query(query, data)
                    else:
                        st.warning("Please enter a query.")
            except ValueError as e:
                st.error(f"Error loading Excel file: {e}")
                logging.error(f"Error loading Excel file: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logging.error(f"Unexpected error: {e}")
    except Exception as e:
        st.error(f"Error initializing ChatGroq LLM: {e}")
        logging.error(f"Error initializing ChatGroq LLM: {e}")

st.sidebar.header("Logs")
st.sidebar.write("Check the logs for detailed debugging information.")
