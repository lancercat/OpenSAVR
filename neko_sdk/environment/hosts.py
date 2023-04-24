import os
# the devices names, user names, ports and addrs are redacted, for reasons

def get_dev_meta(branch="neko_2022_soai_zero"):
    return {
        "dev-1":
        {
            "ltype":"dash",
            "name": "DEV1",
            "username": "Nep",
            "root": os.path.join("/home/Nep/cat/neko_wcki/",branch),
            "port": "9",
            "addr": "md.meows"

        },
        "ABLMAT":
        {
            "ltype": "solid",
            "name":"ABLMAT",
            "username": "Nep",
            "root":os.path.join("/home/Nep/cat/neko_wcki",branch),
            "port": "9",
            "addr": "localhost"
        },
        "dev-2":
        {
            "ltype": "dot",
            "name": "DEV2",
            "username": "Nep",
            "root": os.path.join("/home/Nep/cat/neko_wcki",branch),
            "port": "9",
            "addr": "localhost"
        },
        "dev-3":
            {
                "ltype": "dot",
                "name": "DEV3",
                "username": "Nep",
                "root": os.path.join("/home/Nep/cat/neko_wcki/",branch),
                "port": "9",
                "addr": "localhost"
            },
        "dev-4":
            {
                "ltype": "dot",
                "name": "DEV4",
                "username": "Nep",
                "root": os.path.join("/home/Nep/cat/neko_wcki/", branch),
                "port": "9",
                "addr": "127.0.0.1"
            },
        "dev-5":
            {
                "ltype": "dot",
                "name": "DEV5",
                "username": "Nep",
                "root": os.path.join("/run/media/Nep/data/neko_wcki/", branch),
                "port": "9",
                "addr": "127.0.0.1"
            }
    }
