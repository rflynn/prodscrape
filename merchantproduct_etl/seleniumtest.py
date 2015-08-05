from selenium import webdriver

# re: merchant product ETL...
# explore running javascript via selenium via python
# most merchants these days expose at least as much product info via js as they do in HTML microdata...

# pip install selenium
# brew install phantomjs

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

capabilities = DesiredCapabilities.CHROME
capabilities = DesiredCapabilities.PHANTOMJS
capabilities['loggingPrefs'] = {'browser': 'ALL'}

#driver = webdriver.Chrome(desired_capabilities=capabilities)
driver = webdriver.PhantomJS(desired_capabilities=capabilities)
driver.set_window_size(1120, 550)
#driver.get('https://duckduckgo.com/')
driver.get('http://www.sephora.com/rouge-shine-lipstick-P310714?skuId=1382258')
driver.find_element_by_id('search_form_input_homepage').send_keys("realpython")
driver.find_element_by_id('search_button_homepage').click()
print dir(driver)
print driver.current_url
driver.execute_script('console.log("hello")')
for entry in driver.get_log('browser'):
    print entry
driver.quit()

'''
JSON.parse(document.querySelectorAll('script[seph-json-to-js="sku"]')[0].text).primary_product.display_name
JSON.parse(document.querySelectorAll('script[seph-json-to-js="sku"]')[0].text).primary_product.id
JSON.parse(document.querySelectorAll('script[seph-json-to-js="sku"]')[0].text).sku_number
'''

