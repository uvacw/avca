
# coding: utf-8
## NOTE: This scraper illustrates how to collect images from websites for academic research
## THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



import pickle
import requests
from bs4 import BeautifulSoup
import time
import random
import datetime
import logging
import warnings
import logging as logger
import os
from sqlalchemy import text





def get_links(project, company, brand, main_url_id, url, domain, main_url, collect_links = 0, collect_images = 0, store_html = 0, level=0, links_collected = []):
    print(url)
    logger.info(str(url) + ' collected')
    headers={'User-Agent' : "Mozilla/5.0"}
    try:
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')

        if store_html == 1:
            try:
                os.mkdir(company+'_'+brand)
            except:
                pass
            path = company+'_'+brand+'/'
            filename = url.replace('/','_').replace(':', '_')
            if len(filename) > 200:
                filename = filename[:200]
            with open(path+'html_'+filename+'.html', 'w') as f:
                f.write(page.text)
            

        base = soup.find_all('base', href=True)
        if len(base) > 0:
            base = base[0]['href']
        else:
            base = None


        if collect_links == 1:

            links = []
            if url[-1] == '/':
                url = url[:-1]
            for link in soup.find_all('a', href=True):
                sql = '''INSERT INTO 02_links_unique(project, company, brand, main_url_id, domain, level, link_full, link_source, link_text, link_url, status_followed, from_company) VALUES('''
                link_source = url

                try:
                    link_text = link.contents[0]
                except:
                    link_text = ''
                link_url = link['href']
                link_url = link_url.replace("javascript:window.open('","").replace("','_self')","").replace("')","").replace("'","")

                if link_url.startswith('http'):
                    link_full = link_url

                elif link_url.startswith('/'):
                    link_full = main_url + link_url

                elif link_url.startswith('./'):
                    if link_source.endswith('/'):
                        if base:
                            link_full = base + link_url[2:]
                        else:
                            link_full = link_source + link_url[2:]

                        
                    else:
                        if base:
                            link_full = base + link_url[2:]
                        else:
                            new_source = link_source.split('/')[:-1]
                            new_source = '/'.join(new_source) + '/'
                            link_full = new_source + link_url[2:]

                    
                elif link_url.startswith('#') == False:
                    if link_url.startswith('javascript') == False:
                        if link_url.startswith('whatsapp:') == False:
                            if link_source.endswith('/'):
                                if base:
                                    link_full = base + link_url
                                else:
                                    link_full = link_source + link_url
                            else:
                                if base:
                                    link_full = base + link_url
                                else:
                                    new_source = link_source.split('/')[:-1]
                                    new_source = '/'.join(new_source) + '/'
                                    link_full = new_source + link_url

                if link_full in links_collected:
                    # print(link_full, 'collected already - skipping')
                    pass
                else:

                    if domain in link_full:
                        from_company = '1'
                    else:
                        from_company = '0'
                    
                    
                    status_followed = '0'

                    sql += '''"'''+ project + '''", '''
                    sql += '''"'''+ company + '''", '''
                    sql += '''"'''+ brand + '''", '''
                    sql += '''"'''+ str(main_url_id) + '''", '''
                    sql += '''"'''+ domain + '''", '''
                    sql += '''"'''+ str(level) + '''", '''
                    sql += '''"'''+ link_full + '''", '''
                    sql += '''"'''+ link_source + '''", '''
                    sql += '''"'''+ str(link_text) + '''", '''
                    sql += '''"'''+ link_url + '''", '''
                    sql += '''"'''+ status_followed + '''", '''
                    sql += '''"'''+ from_company + '''")'''

                    try:
                        con.execute(sql)
                    except Exception as e:
                        try:
                            sql = '''INSERT INTO 02_links(project, company, brand, main_url_id, domain, level, link_full, link_source, link_text, link_url, status_followed, from_company) VALUES('''
                            sql += '''"'''+ project + '''", '''
                            sql += '''"'''+ company + '''", '''
                            sql += '''"'''+ brand + '''", '''
                            sql += '''"'''+ str(main_url_id) + '''", '''
                            sql += '''"'''+ domain + '''", '''
                            sql += '''"'''+ str(level) + '''", '''
                            sql += '''"'''+ link_full + '''", '''
                            sql += '''"'''+ link_source + '''", '''
                            sql += '''"'''+ 'error_link_text' + '''", '''
                            sql += '''"'''+ link_url + '''", '''
                            sql += '''"'''+ status_followed + '''", '''
                            sql += '''"'''+ from_company + '''")'''
                            logger.info(str(e))
                            logger.info(sql)
                        except Exception as e:
                            logger.info(str(e))
                            logger.info(sql)


        if collect_images == 1:
            pics = soup.find_all('img')
            for pic in pics:
                width = pic.get('width', 0)
                height = pic.get('height', 0)
                alt_text = pic.get('alt', '')
                link_url = pic.get('src', '')
                link_source = url
                if link_url.startswith('/'):
                    img_link_full = main_url + link_url
                else:
                    img_link_full = link_url
                status_downloaded = '0'

                sql = '''INSERT INTO 03_images(project, company, brand, main_url_id, domain, level, link_full, link_source, link_url, status_downloaded, image_height, image_width, image_alt) VALUES('''

                sql += '''"'''+ project + '''", '''
                sql += '''"'''+ company + '''", '''
                sql += '''"'''+ brand + '''", '''
                sql += '''"'''+ str(main_url_id) + '''", '''
                sql += '''"'''+ domain + '''", '''
                sql += '''"'''+ str(level) + '''", '''
                sql += '''"'''+ img_link_full + '''", '''
                sql += '''"'''+ link_source + '''", '''
                sql += '''"'''+ link_url + '''", '''
                sql += '''"'''+ status_downloaded + '''", '''
                sql += str(width) + ''', '''
                sql += str(height) + ''', '''
                sql += '''"'''+ str(alt_text) + '''")'''
                # print(sql)

                try:
                    con.execute(sql)
                except Exception as e:
                    try:

                        sql = '''INSERT INTO 03_images(project, company, brand, main_url_id, domain, level, link_full, link_source, link_url, status_downloaded, image_height, image_width, image_alt) VALUES('''
                        sql += '''"'''+ project + '''", '''
                        sql += '''"'''+ company + '''", '''
                        sql += '''"'''+ brand + '''", '''
                        sql += '''"'''+ str(main_url_id) + '''", '''
                        sql += '''"'''+ domain + '''", '''
                        sql += '''"'''+ str(level) + '''", '''
                        sql += '''"'''+ link_full + '''", '''
                        sql += '''"'''+ link_source + '''", '''
                        sql += '''"'''+ img_link_full + '''", '''
                        sql += '''"'''+ status_downloaded + '''", '''
                        sql += '''"'''+ str(width) + '''", '''
                        sql += '''"'''+ str(height) + '''", '''
                        sql += '''"'''+ str('error') + '''")'''
                        con.execute(sql)
                    except Exception as e:
                        logger.info(str(e))
                        logger.info(sql)



        
        time.sleep(random.uniform(0.5,5))
    except Exception as e:
        logger.info('error retrieving URL')
        logger.info(str(url))
        logger.info(str(e))
        
    return 
     



def run_scraper(project, company, brand, main_url_id, url, domain, main_url, collect_links = 0, collect_images = 0, store_html = 0, levels = 1, skip_level0=False):
    
    if skip_level0 == False:
        links_collected = get_links_collected(project, company, brand, status_followed = None, from_company = None)

        level = 0

        get_links(project, company, brand, main_url_id, url, domain, main_url, collect_links = collect_links, collect_images = collect_images, store_html = store_html, level = level)

        sql = '''UPDATE 02_links SET status_followed = 1 WHERE link_full = "''' + url + '''"'''
        con.execute(sql)
    else:
        sql = '''SELECT level FROM 02_links WHERE project ="''' + project + '''" AND company = "'''+ company + '''" AND brand = "'''+ brand + '''" ORDER BY level DESC limit 1'''
        res_levels = con.execute(sql)
        level = 0
        if res_levels[0][0] > 0:
            print('resuming at level', res_levels[0][0])
            level = res_levels[0][0] -1




    links_to_collect = get_links_collected(project, company, brand, status_followed = 0, from_company = 1)
    links_collected = get_links_collected(project, company, brand, status_followed = None, from_company = None)



    # In[ ]:


    while level < levels:
        links_to_collect = get_links_collected(project, company, brand, status_followed = 0, from_company = 1)
        logger.info(str('links_to_collect: ' + str(links_to_collect)))
        for link_full in links_to_collect:
            links_collected = get_links_collected(project, company, brand, status_followed = 1, from_company = None)
            logger.info(str('links_collected: ' + str(links_collected)))
            try:
                
                if link_full not in links_collected:
                    if link_full.endswith('.pdf'):
                        logger.info(str(link_full + ' skipped: PDF'))
                    elif 'mailto:' in link_full:
                        logger.info(str(link_full + ' skipped: email'))
                    elif link_full.endswith('.exe'):
                        logger.info(str(link_full + ' skipped: EXE'))
                    
                    else:
                        get_links(project, company, brand, main_url_id, link_full, domain, main_url, collect_links = collect_links, collect_images = collect_images, store_html = store_html, level = level + 1)
                        sql = '''UPDATE 02_links SET status_followed = 1 WHERE link_full = "''' + link_full + '''"'''
                        con.execute(sql)

                else:
                    logger.info(str(link_full + ' skipped: already collected'))
            except Exception as e:
                log_error(link_full, str(e))
                logger.info(str(link_full + ' error'))
                


        level += 1
        print('level', level, 'completed')






def get_pending_company(project = None):
    if project:
        sql = '''SELECT id, project, company, brand, status, store_html, collect_links, collect_images, link_full, main_url, domain, levels, last_level_complete FROM 01_to_get WHERE status = "pending" AND levels - last_level_complete > 0 AND project = "'''+ project+'''" LIMIT 1'''

    else:
        sql = '''SELECT id, project, company, brand, status, store_html, collect_links, collect_images, link_full, main_url, domain, levels, last_level_complete FROM 01_to_get WHERE status = "pending" AND levels - last_level_complete > 0 LIMIT 1'''
    res = con.execute(sql)
    return res.cursor.fetchall() 


def update_status_url(id, status):
    sql = '''UPDATE 01_to_get SET status = "'''+ status+'''" WHERE id = ''' + str(id)
    con.execute(sql)

def get_links_collected(project, company, brand, status_followed = None, from_company = None):
    if status_followed == None:
        if from_company == None:
            sql = '''SELECT link_full FROM 02_links WHERE project = "''' + project + '''" AND company = "''' + company + '''" AND brand = "'''+brand+'''"'''
        else:
            sql = '''SELECT link_full FROM 02_links WHERE project = "''' + project + '''" AND company = "''' + company + '''" AND brand = "'''+brand+'''" AND from_company = ''' + str(from_company)
    else:
        if from_company == None:
            sql = '''SELECT link_full FROM 02_links WHERE status_followed = ''' + str(status_followed) + ''' AND project = "''' + project + '''" AND company = "''' + company + '''" AND brand = "'''+brand+'''"'''
        else:
            sql = '''SELECT link_full FROM 02_links WHERE status_followed = ''' + str(status_followed) + ''' AND project = "''' + project + '''" AND company = "''' + company + '''" AND brand = "'''+brand+'''" AND from_company = ''' + str(from_company)

    res = con.execute(sql)
    res = [item[0] for item in res.cursor.fetchall()]
    return res


def get_pending_links(project, company, brand, level):
    sql = '''SELECT link_full FROM 02_links_unique WHERE project = '{project}' AND company = '{company}' AND brand = '{brand}' AND level = {level} AND status_followed = 0 AND from_company = 1'''.format(**locals())

    res = con.execute(sql)
    res = [item[0] for item in res.cursor.fetchall()]
    return res

def get_collected_links(project, company, brand):
    sql = '''SELECT link_source FROM 02_links_unique WHERE project = '{project}' AND company = '{company}' AND brand = '{brand}' '''.format(**locals())
    res = con.execute(sql)
    res = [item[0] for item in res.cursor.fetchall()]

    sql = '''SELECT link_full FROM 02_links_unique WHERE project = '{project}' AND company = '{company}' AND brand = '{brand}' '''.format(**locals())
    res2 = con.execute(sql)
    res2 = [item[0] for item in res2.cursor.fetchall()]

    return list(set(res + res2))


def log_error(link_full, error):
    pass


def process_scraper():
    from db_alchemy_scraper import con
    global con
    try:
        logger.basicConfig(filename=str('log_' + str(datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")))+'.log', level=logger.INFO)
        pending = get_pending_company()
        if len(pending) > 0:
            id, project, company, brand, status, store_html, collect_links, collect_images, link_full, main_url, domain, levels, last_level_complete = pending[0]  
            logger.info('{id}, {project}, {company} levels {levels} last_level_complete: {last_level_complete}'.format(**locals()))
            links_to_collect = get_pending_links(project, company, brand, last_level_complete)
            if len(links_to_collect) == 0:
                print('no links to collect, adding main link to be sure')
                links_to_collect.append(link_full)
            
            update_status_url(id, 'ongoing')

            # print(links_collected)
            levelnew= last_level_complete + 1

            for link_to_collect in links_to_collect:
                if link_to_collect.endswith('.pdf'):
                    logger.info(str(link_full + ' skipped: PDF'))
                elif 'mailto:' in link_to_collect:
                    logger.info(str(link_to_collect + ' skipped: email'))
                elif link_to_collect.endswith('.exe'):
                    logger.info(str(link_to_collect + ' skipped: EXE'))
                elif link_to_collect.startswith('javascript'):
                    logger.info(str(link_to_collect + ' skipped: java'))

                else:
                    link_to_collect = link_to_collect.replace("'","")
                    links_collected = get_collected_links(project, company, brand)
                    # print(link_to_collect)
                    
                    get_links(project, company, brand, id, link_to_collect, domain, main_url, collect_links = collect_links, collect_images = collect_images, store_html = store_html, level=levelnew, links_collected = links_collected)
                    con.execute('''UPDATE 02_links_unique SET status_followed = 1 WHERE link_full = '{link_to_collect}' '''.format(**locals()))
                    print('{link_to_collect} completed'.format(**locals()))
                    links_collected.append(link_to_collect)
                    links_to_collect.remove(link_to_collect)
                    total_collected = len(links_collected)
                    total_to_collect = len(links_to_collect)
                    print('completed {link_to_collect} - total links collected = {total_collected}, total links to be collected at this level = {total_to_collect}'.format(**locals()))
            logger.info('''{company} {brand} level {levelnew} completed'''.format(**locals()))
            con.execute('''UPDATE 01_to_get SET last_level_complete = {levelnew} WHERE project = "{project}" AND company = "{company}" AND brand = "{brand}" '''.format(**locals()))

            update_status_url(id, 'pending')
        else:
            print('nothing to do?')
            logger.info('nothing to do?')
    except Exception as e:
        print(link_to_collect)
        print(e)
        logger.info('failed')
        logger.info(str(e))
        update_status_url(id, 'failed')

    return 


if __name__ == "__main__":
    process_scraper()



    



    # 
    # logger.info(str('started with ' + main_url + ' for company ' +  company + ' for project ' + project))
    # logger.info(str('collect_images = ' + str(collect_images) + ' collect_links = ' + str(collect_images) + ' store_html = ' + str(store_html) + ' levels to crawl = ' + str(levels)))

    # run_scraper(project, company, brand, id, link_full, domain, main_url, collect_links = collect_links, collect_images = collect_images, store_html = store_html, levels = levels)
    # update_status_url(id, 'completed')



