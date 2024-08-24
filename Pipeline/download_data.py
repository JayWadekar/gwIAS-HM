#%%
import json, urllib, os
import utils


gps_start_o2 = 1164556817
gps_end_o2 = 1187733618


# Works with the latest version of GWOSC
def get_url_list_new(detector = 'H1', GPSstart = 1126051217, GPSend = 1137254417):
    locate = utils.load_module('gwosc.locate')
    fnames = locate.get_urls(detector=detector, start=GPSstart, end=GPSend)
    return fnames


# Does not work with the latest version of GWOSC
def get_url_list(detector = 'H1', dataset = 'O1', GPSstart = 1126051217, GPSend   = 1137254417):

    url_json_format = 'https://gw-openscience.org/archive/links/{0}/{1}/{2}/{3}/json/'

    url = url_json_format.format(dataset, detector, GPSstart, GPSend)
    print("Tile catalog URL is ", url)

    r = urllib.request.urlopen(url).read()    # get the list of files
    tiles = json.loads(r)             # parse the json

    print(tiles['dataset'])
    print(tiles['GPSstart'])
    print(tiles['GPSend'])
    url_list = [x['url'] for x in tiles['strain'] if x['format'] == 'hdf5']
    return url_list


#Implicitly using their website structure to get the list of files.
def gen_url_list(detector_magic = 'L-L1', gps_start = gps_start_o2, gps_end = 1187733618):

    big_gps_start_O2 = 1163919360
    n_files = 256
    filelen = 4096

    urls = []
    for cur_big_gps in range(big_gps_start_O2, gps_end, filelen*n_files):
        for small_gps_ind in range(n_files):
            small_gps = cur_big_gps + small_gps_ind * filelen
            if (gps_start-filelen) <small_gps < gps_end :
                urls.append(f'https://www.gw-openscience.org/archive/data/O2_4KHZ_R1/{cur_big_gps}/{detector_magic}_GWOSC_O2_4KHZ_R1-{small_gps}-4096.hdf5')

    return urls

def download_all_files(url_list, dirname):
    for url in url_list:
        fname = os.path.join(dirname, url.split('/')[-1])
        try:
            response = urllib.request.urlopen(url)
            st = response.read()
        except:
            print("Failed: ", url)
            continue

        with open(fname,'wb') as f:
            f.write(st)
            print ("Written ", url ," Successfully")
