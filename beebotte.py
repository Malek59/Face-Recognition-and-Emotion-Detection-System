

from beebotte import *
import datetime

bbt = bbt("4U3w4o5dznnH5N6Stb0v0ABW", "FbFWidZknEouBMbXtAzkHDvkAM46nmjx")

current_time = datetime.datetime.now()
date = current_time.date().strftime('%Y-%m-%d')
time = current_time.time().strftime('%H:%M:%S')

                    message = {

                        "Name": id,
                        "Emotion": emotion,
                        "Date": date,
                        "Time": time
                    }

                    bbt.write('Data', 'Name', id)
                    bbt.write('Data', 'Emotion', emotion)
                    bbt.write('Data', 'Date', date)
                    bbt.write('Data', 'Time', time)




