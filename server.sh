#!/usr/bin/sh -x
python3 server.py &
bg
disown -h
