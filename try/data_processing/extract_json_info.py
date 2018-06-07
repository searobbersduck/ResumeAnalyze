# !/usr/bin/env python2
# -*- coding:utf-8 -*-

# 字段参考页面：http://conf.jihui.in/pages/viewpage.action?pageId=5275777

import json

raw_json_file = '/Users/higgs/beast/code/work/ResumeAnalyze/try/data/resume_json/5465.doc_json.txt'

json_raw = None

with open(raw_json_file) as f:
    json_raw = json.load(f)

print(json_raw)