261114
Переписываю мультипроцессинг, дабы исключить:
1. Застои пула, в ожидании последних батчей обработки
2. Постоянную генерацию с нуля пула процессов (актуально для огромного кол-ва малых файлов)
3. Простой обработки, когда идёт дозапись результата перед переходом на новый файл

Что было решено сделать:
Создание класса MultiProcessorManager, отвечающий за непрерывное использование единожды созданного пула процессов и организации параллельной потоковой записи и создания батчей на свободном ядре/ядрах, дабы исключить и простаивание единственного ядра.
Упрощённая схема:
```mermaid

flowchart LR
	subgraph f1["File_1"]
	end
	subgraph f2['File_2']
	end
	subgraph f3['File_3']
	end
	
	subgraph rf1["resulted_data_1"]
	end
	subgraph rf2["resulted_data_2"]
	end
	subgraph rf3["resulted_data_3"]
	end
   subgraph st["start()"]
   end
   subgraph batch_func['batch_function']
   end
	subgraph MPMan["MultiProcessorManager"]
		subgraph tqdm_main["tqdm main processed file counter"]
		end
		subgraph tqdm_sec["tqdm file batch counter"]
		end
		subgraph ids["очередь free_ids <br/> Контроль за одновременной подготовкой батчей заранее (не больше 3) "]
      end

      subgraph meta["словарь meta для записи результатов"]
      end
      subgraph THR_Batcher["Поток батчей"]

		end
		subgraph batch["очередь batches"]
		end
		subgraph batch_gen["Генератор батчей из очереди"]
		end

		subgraph THR_Writer["Main поток, записывающий результат с распределением по id"]
		end
		subgraph THR_Pool["Pool processing"]
		end
	end

	%% wr_func["Функция записи результатов"]

	%% wr_func --> THR_Writer
	
	f1 --> batch_func
	f2 --> batch_func
	f3 --> batch_func
	THR_Writer ---> rf1
	THR_Writer ---> rf2
	THR_Writer ---> rf3
	st -.-> |"Заполнение свободными индексами (0,1,2)"|ids
	ids --> |"get()"| THR_Batcher
   batch_func --> THR_Batcher
   THR_Batcher --> meta
   THR_Batcher --> batch
   meta --> THR_Writer
   batch --> batch_gen
   batch_gen --> THR_Pool
   THR_Writer --> tqdm_sec
   tqdm_sec --> tqdm_main
   tqdm_sec -.-> |"внесение в очередь освободившихся ids, после последнего батча (n processed = total batches)
   put(id)"|ids
	%% THR_Batcher --> THR_Pool
	THR_Pool --> THR_Writer
	%% THR_Pool -.-> |"terminate file processing signal"|THR_Writer
%%    THR_Writer -.-> |"внесение в очередь освободившихся ids
%%    put(id)"| ids
```

Теперь стоит задача сделать тоже самое для estimate_peak_density_kde, стараясь сохранить функцию _compute_kde:

```mermaid
flowchart LR
	data["datasource"]
	subgraph kde_old["_compute_kde(old)"]
		pkl["peaklist"]
		bandwidth["bandwidth computations"]
		bandwidth --> |add column| pkl
		spl["split data by sigma distance gap"]
		spl_data["splitted data (bathces)"]
		pkl --> spl
		spl --> spl_data
		pool["process pool"]
		spl_data --> pool
		kdx["kde mz plot"]
		kdy["kde intensity plot"]
		pool --> kdx
		pool --> kdy
		
	end
	kdx --> hdf5
	kdy --> hdf5
	data --> pkl
```
```mermaid
flowchart LR
	ds["DataSet"]
	subgraph kde_new["new _compute_kde"]
		subgraph smpl["sample extraction thread"]
			pkl2["peaklist"]
			bandwidth2["bandwidth computations"]
			bandwidth2 --> |add column| pkl2
			spl2["split data by sigma distance gap"]
			
			pkl2 --> spl2


		end
		spl_data2["splitted data (bathces)"]
					spl2 --> spl_data2
		pool2["process pool"]
		spl_data2 --> pool2
		kdx2["kde mz plot"]
		kdy2["kde intensity plot"]
		pool2 --> kdx2
		pool2 --> kdy2
	end
	ds --> |from one sample|pkl2
	kdx2 --> |yield|hdf52["hdf5 sample"]
	kdy2 --> |yield|hdf52
```