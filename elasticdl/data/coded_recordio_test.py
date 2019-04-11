import unittest
import codecs
import tempfile

from .coded_recordio import File

encoder = lambda x: codecs.encode(x, "rot_13").encode()
decoder = lambda x: codecs.decode(x.decode(), "rot_13")
DATA = [
    "china",
    "usa",
    "russia",
    "india",
    "thailand",
    "finland",
    "france",
    "germany",
    "poland",
    "san marino",
    "sweden",
    "neuseeland",
    "argentina",
    "canada",
    "ottawa",
    "bogota",
    "panama",
    "united states",
    "brazil",
    "barbados",
]


class CodedRecordIOTest(unittest.TestCase):
    def testAll(self):
        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        # Write an encoded RecordIO file.
        with File(tmp_file.name, "w", encoder=encoder) as coded_w:
            for data in DATA:
                coded_w.write(data)

        # Verify raw content
        with File(tmp_file.name, "r") as raw_r:
            rlist = [raw_r.get(i) for i in range(len(DATA))]
            self.assertEqual(list(map(encoder, DATA)), rlist)

        # Verify decoded content, with get() interface.
        with File(tmp_file.name, "r", decoder=decoder) as coded_r:
            rlist = [coded_r.get(i) for i in range(len(DATA))]
            self.assertEqual(DATA, rlist)

        # Verify decoded content, with iterator interface 
        with File(tmp_file.name, "r", decoder=decoder) as coded_r:
            rlist = list(coded_r)
            self.assertEqual(DATA, rlist)

if __name__ == "__main__":
    unittest.main()
