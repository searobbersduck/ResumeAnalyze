#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import os
import time
import string
import pbjson
import json as simplejson
import json
import resume_pb2
import re

def main():
    res = None
    with open('/Users/higgs/beast/code/work/ResumeAnalyze/try/data/resume_json/5465.doc_json.txt') as f:
        res = simplejson.load(f)
    t = pbjson.dict2pb(resume_pb2.Resume, res)

    resume_cont = ""
    resume_cont_json = None
    with open('/Users/higgs/beast/code/work/ResumeAnalyze/try/data/resume_json/33526.doc_origin.txt') as f:
        resume_cont_json = json.load(f)

    for cont in resume_cont_json:
        resume_cont += "\n"
        resume_cont += cont['text']

    # 提取解析的标签中的每一项，在原文中（即resume_cont_json）进行匹配，这里以教育经历中的学校为例
    school = t.educationExperiences[0].school.title
    school_pos = [m.start() for m in re.finditer(school, resume_cont)]
    print(school_pos)
    chineseName = t.chineseName
    chineseName_pos = [m.start() for m in re.finditer(chineseName, resume_cont)]
    print(chineseName_pos)
    privateEmail = t.privateEmail
    privateEmail_pos = [m.start() for m in re.finditer(privateEmail, resume_cont)]
    print(privateEmail_pos)





if __name__  ==  "__main__":
    main()