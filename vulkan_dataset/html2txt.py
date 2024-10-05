from bs4 import BeautifulSoup

# Open the HTML file
with open("vulkan_spec.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Extract the text
text = soup.get_text()

# Save the text to a .txt file
with open("vulkan_spec.txt", "w", encoding="utf-8") as text_file:
    text_file.write(text)
