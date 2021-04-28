from modules.search.Search import Search


def searchWorker(pause, exitRequest, pipelineIn, pipelineOut):
  
    print("Start Search")
    
    search = Search()
    
    pause.acquire()
    
    values = pipelineIn.get()
    search_result = search.perform_search(values["tentGPS"], values["planeGPS"], values["angle"])
    pipelineOut.put(search_result)
    
    pause.release()
