from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import logging
import time
import datetime

logging.basicConfig(filename="../../Downloads/rabbitmonitor/test.log", level=logging.ERROR)
ignored_exceptions=(StaleElementReferenceException,)
chrome_options = Options()
# chrome_options.add_argument("--headless")
br = webdriver.Chrome("./chromedriver", options=chrome_options)

while True:
    br.get("https://aavts.gpsfinderpro.com/Index.html")
    username = WebDriverWait(br, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
    username.send_keys("admin")
    br.find_element_by_id("password").send_keys("2#3V1i0s8i8o3n1")
    br.execute_script("arguments[0].click();", br.find_element_by_id("signInBtn"))
    search = WebDriverWait(br, 10).until(
            EC.element_to_be_clickable((By.ID, "select2-appsDropdown-container"))
        )
    search.click()
    br.find_element_by_xpath("//ul[@id='select2-appsDropdown-results']/li[last()]").click()
    br.find_element_by_id("btnGoToApp").click()

    integrations = WebDriverWait(br, 10).until(
            EC.element_to_be_clickable((By.ID, "DataListTabs_LinkButtonTab_11"))
        )
    integrations.click()
    while True:
        try:
            while True:
                try:
                    backlog = WebDriverWait(br, 5, ignored_exceptions=ignored_exceptions).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="RabbitMQContainer"]/div[7]/fieldset/div/div[2]/div[last()-2]/div[1]'))
                    )
                    backlog_text = backlog.text
                    break
                except StaleElementReferenceException:
                    continue

            logging.warning("%s: %s" %(datetime.datetime.now(), backlog_text))
            if int(backlog_text) > 1000:
                logging.error("Backlog: " + backlog_text)
                br.find_element_by_xpath('//*[@id="RabbitMQContainer"]/div[4]/fieldset/div[2]/table/tbody/tr[2]/td[11]/button').click()
                br.find_element_by_xpath('//*[@id="RabbitMQContainer"]/div[3]/fieldset/div[12]/div[2]/button[1]').click()
                logging.error("Updated at " + str(datetime.datetime.now()))
            time.sleep(1)
        except Exception as e:
            logging.error(e)
            break
