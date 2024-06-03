
import os
import logging as log
import urllib


class Table(dict):
    def __init__(self, init=None, safe=True):
        if not init:
            init = {}
        super().__init__(init)
        self.__dict__['safe'] = safe  # Explicitly define safe in the object's dictionary

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        elif attr == 'safe':
            return self.__dict__.get('safe', False)  # Access 'safe' without triggering __getattr__
        else:
            if self.__dict__.get('safe', False):
                return None
            else:
                raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __setattr__(self, attr, data):
        if attr == 'safe':
            self.__dict__[attr] = data
            return

        self[attr] = data
        if data is None:
            del self[attr]


def tablify(output, safe=True):
    if isinstance(output, dict):
        data = {}
        for n in output:
            data[n] = tablify(output[n])
        return Table(data, safe)

    elif isinstance(output, list):
        return [tablify(i) for i in output]

    else:
        return output


def read_file(res_file, res_opt=False):
    if not os.path.isfile(res_file):
        if not res_opt:
            log.warning("Invalid/Missing file resource: %s", res_file)
        return None

    with open(res_file, "r", encoding="utf-8") as f:
        res_value = f.read()

    return res_value


def write_file(res_file, res_data=None):
    if not res_data:
        try:
            os.remove(res_file)
        except:
            pass
        return

    with open(res_file, "w", encoding="utf-8") as f:
        f.write(res_data)


def download_file(res_url, res_file):
    retry_cnt = 0
    retry_lim = 3
    while retry_cnt < retry_lim:
        retry_cnt += 1
        try:
            urllib.request.urlretrieve(res_url, res_file)
            return True
        except Exception as e:
            log.warning("Failed %i/%i to get image '%s': %s",
                     retry_cnt, retry_lim, res_url, e)

    return False
