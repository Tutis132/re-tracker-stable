import os
import sys
import logging
import argparse
import configparser

debug = bool(os.getenv('DEBUG'))
onerun = None
dryrun = None
recheck = None
report = None
target = None
table = None
url = None
type = None

if debug:
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('target', nargs='?')
parser.add_argument('-r', action='store_true', dest='recheck')
parser.add_argument('-p', action='store_true', dest='report')

config = configparser.ConfigParser()
config.read('estate.conf')

def load_section(section_name):
    global target, table, url, type
    if section_name in config:
        section = config[section_name]
        target = section_name
        table = section.get('table')
        url = section.get('url')
        type = section.get('type')
    else:
        raise ValueError(f"Section '{section_name}' not found in the configuration.")

class Params:
    def __init__(self, section=None):
        if section:
            load_section(section)

    def __getattr__(self, name):
        global_values = globals()
        if name in global_values:
            return global_values[name]
        if not hasattr(self, name):
            self.resolve_dynamic(name)
        return getattr(self, name)

    def resolve_dynamic(self, name):
        if name == 'target' or name == 'table' or name == 'url' or name == 'type':
            self.resolve_target()
        elif name in ['onerun', 'dryrun', 'debug', 'recheck', 'report']:
            attr = f"resolve_{name}"
            if hasattr(self, attr):
                setattr(self, name, getattr(self, attr)())

    def resolve_target(self):
        global target, table, url, type
        if target is None:
            args = parser.parse_args()
            target = args.target or os.getenv('TARGET')
        
        if target and target in config:
            section = config[target]
            table = section.get('table')
            url = section.get('url')
            type = section.get('type')
        elif not table or not url or not type:
            raise ValueError("Configuration for the target is incomplete or missing.")

    @staticmethod
    def resolve_onerun():
        global onerun
        onerun = os.getenv('ONE_RUN')
        return onerun

    @staticmethod
    def resolve_dryrun():
        global dryrun
        dryrun = os.getenv('DRY_RUN')
        return dryrun

    @staticmethod
    def resolve_debug():
        global debug
        debug = bool(os.getenv('DEBUG'))
        return debug

    @staticmethod
    def resolve_recheck():
        global recheck
        args = parser.parse_args()
        recheck = args.recheck
        return recheck

    @staticmethod
    def resolve_report():
        global report
        args = parser.parse_args()
        report = args.report
        return report

    def reset(self):
        global onerun, dryrun, debug, recheck, report, target, table, url, type
        onerun = dryrun = debug = recheck = report = target = table = url = type = None

# Replace the module object in sys.modules with an instance of Params
sys.modules[__name__] = Params()
