from modules.communications.FltConnSendWorker import flt_conn_send_worker
import multiprocessing as mp

print('Started')
def dummy_producer(pipelineIn):
    	data = {
    	"type": "QR_COORDINATES",
    	"data": {
    	"longitude": 1.23,
    	"latitude": 1.23,
    	"qr_scan_flag": 1,
    	"takeoff_command": 0,
    	"detect_flag": 1
    	}
    	}
    	pipelineIn.put(data)

dummy_messages_pipeline = mp.Queue()
pause = mp.Lock()
quit = mp.Queue()
dummy_messages_producer = mp.Process(target=dummy_producer, args=(dummy_messages_pipeline,))
flt_conn_send_consumer = mp.Process(target=flt_conn_send_worker, args=(pause, quit,dummy_messages_pipeline))

processes = [dummy_messages_producer, flt_conn_send_consumer]

for p in processes:
	p.start()
print('Started all processes')
