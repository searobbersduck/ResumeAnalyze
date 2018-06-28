#python resume_extract_try.py --char_vector_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vec.txt --char_vocab_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vocab.txt

#python resume_extract_try.py --char_vector_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vec.txt --char_vocab_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vocab.txt --log_dir log_except_workexpr --tfdata ./tfdata_except_workexpr2

#python resume_extract_try.py --char_vector_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vec.txt --char_vocab_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vocab.txt --log_dir log_basicinfo_edu --tfdata ./tfdata_basicinfo_edu

#python resume_extract_try.py --char_vector_file ../../extsrc/ttt_vec.txt --char_vocab_file ../../extsrc/ttt_vocab.txt --log_dir log_all --tfdata ./tfdata_all --batch_size 4 --lr 5e-5
# debug --char_vector_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vec.txt --char_vocab_file /Users/higgs/beast/code/work/ResumeAnalyze/extsrc/ttt_vocab.txt --log_dir log_all --tfdata ./tfdata_all

python resume_extract_try.py --char_vector_file ../../extsrc/ttt_vec.txt --char_vocab_file ../../extsrc/ttt_vocab.txt --log_dir log_all_basicinfo --tfdata ./tfdata_all0 --batch_size 4 --lr 1e-4