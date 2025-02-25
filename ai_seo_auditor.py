import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import re
import logging
import urllib.parse
from huggingface_hub import InferenceClient
import base64
import os

# Optional: for readability analysis; install via pip install textstat
try:
    import textstat
    READABILITY_ENABLED = True
except ImportError:
    READABILITY_ENABLED = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === CONFIGURATION ===

# Hugging Face API configuration
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

if HF_API_TOKEN is None:
    print("Error: Hugging Face token not found in environment variables. Please set the HUGGINGFACE_TOKEN environment variable.")
    st.error("Hugging Face token not found in environment variables. Please set the HUGGINGFACE_TOKEN environment variable.")
    st.stop() #Stop the streamlit app from running.
else:
    print("Hugging face token loaded successfully")

HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Timeout for HTTP requests (in seconds)
REQUEST_TIMEOUT = 10
# === Helper Functions ===

def get_base_url_from_html(page_html):
    """Extracts the base URL from the provided HTML."""
    soup = BeautifulSoup(page_html, "html.parser")
    base_tag = soup.find("base")
    if base_tag and base_tag.get("href"):
        return base_tag["href"]
    else:
        # If no <base> tag, try to extract from the first link's absolute URL, or return None
        first_link = soup.find("a", href=True)
        if first_link and urllib.parse.urlparse(first_link["href"]).netloc:
            return urllib.parse.urljoin(first_link["href"], "/")
        return "" #Return empty string if base url cannot be found

def parse_html(page_html, base_url):
    soup = BeautifulSoup(page_html, "html.parser")
    meta_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta_description = ""
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag:
        meta_description = meta_desc_tag.get("content", "").strip()
    headers = {f"h{level}": len(soup.find_all(f"h{level}")) for level in range(1, 7)}
    images = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "").strip()
        full_src = urllib.parse.urljoin(base_url, src)
        images.append({"src": full_src, "alt": img.get("alt", "No alt text provided").strip()})
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        full_href = urllib.parse.urljoin(base_url, href)
        try:
            response = requests.head(full_href, allow_redirects=True, timeout=REQUEST_TIMEOUT)
            if response.status_code >= 400:
                response = requests.get(full_href, timeout=REQUEST_TIMEOUT)
            status_code = response.status_code
        except requests.exceptions.RequestException:
            status_code = "Error"
        link_type = "internal" if urllib.parse.urlparse(full_href).netloc == urllib.parse.urlparse(base_url).netloc else "external"
        links.append({"url": full_href, "status": status_code, "type": link_type})
    for element in soup(["script", "style"]):
        element.extract()
    main_text = soup.get_text(separator=" ")
    main_text = re.sub(r'\s+', ' ', main_text).strip()
    return {"meta_title": meta_title, "meta_description": meta_description, "headers": headers, "images": images, "links": links, "main_text": main_text}

def generate_audit_report(url, parsed_data):
    report = {"url": url, "meta_title": parsed_data["meta_title"], "meta_description": parsed_data["meta_description"], "headers": parsed_data["headers"], "image_count": len(parsed_data["images"]), "link_count": len(parsed_data["links"]), "internal_link_count": len([link for link in parsed_data["links"] if link["type"] == "internal"]), "external_link_count": len([link for link in parsed_data["links"] if link["type"] == "external"]), "broken_link_count": len([link for link in parsed_data["links"] if link["status"] != 200 and link["status"] != "Error"]), "error_link_count": len([link for link in parsed_data["links"] if link["status"] == "Error"]), "main_text_length": len(parsed_data["main_text"])}
    return report

def send_to_mistral(report, hf_url, hf_api_token):
    client = InferenceClient(model=hf_url, token=hf_api_token)
    prompt = ("You are an SEO expert. Analyze the following SEO audit report and provide recommendations for improvement:\n\n" f"{json.dumps(report, indent=4)}")
    try:
        logging.info("Sending audit report to Mistral...")
        response = client.text_generation(prompt)
        if isinstance(response, str):
            return response.strip()
        elif isinstance(response, list) and response and "generated_text" in response[0]:
            return response[0]["generated_text"].strip()
        else:
            logging.error(f"Unexpected API response: {response}")
            return "Failed to parse Mistral response."
    except Exception as e:
        logging.error(f"Error communicating with Mistral: {e}")
        return "Error connecting to Mistral."

def create_download_link(val, filename):
    b64 = base64.b64encode(val.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Report</a>'

# === Streamlit App ===

st.title("SEO Audit Tool")

uploaded_file = st.file_uploader("Upload scraped JSON file", type=["json"])

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
    except Exception as e:
        st.error(f"Error reading JSON file: {e}")
        st.stop()

    if not isinstance(data, (list, dict)):
        st.error("Uploaded JSON file must be a list or a dictionary.")
        st.stop()

    if isinstance(data, list):
        valid_articles = [article for article in data if isinstance(article, dict) and "page_html" in article]
        if not valid_articles:
            st.error("Uploaded JSON file does not contain any valid articles with a 'page_html' field.")
            st.stop()
    elif isinstance(data, dict) and "page_html" not in data:
        st.error("Uploaded JSON file does not contain a 'page_html' field.")
        st.stop()
    else:
        valid_articles = [data]

    st.subheader("Audit Reports")

    for i, article in enumerate(valid_articles):
        st.write(f"### Article {i+1}")
        html = article["page_html"]
        base_url = get_base_url_from_html(html) #Auto get the base url
        parsed_data = parse_html(html, base_url)
        report = generate_audit_report(base_url, parsed_data)
        st.json(report)
        enhanced_report = send_to_mistral(report, HF_URL, HF_API_TOKEN)
        if enhanced_report:
            with st.expander("Enhanced Analysis"):
                st.write(enhanced_report)
            report_text = f"URL: {base_url}\n\nAudit Report:\n{json.dumps(report, indent=4)}\n\nEnhanced Analysis:\n{enhanced_report}"
            download_link = create_download_link(report_text, f"audit_report_{i+1}.txt")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.warning("Failed to retrieve enhanced analysis.")
            report_text = f"URL: {base_url}\n\nAudit Report:\n{json.dumps(report, indent=4)}"
            download_download_link = create_download_link(report_text, f"audit_report_{i+1}.txt")
            st.markdown(download_link, unsafe_allow_html=True)

else:
    st.warning("Please upload a JSON file to continue.")
