for alpha in '0.3' '0.5';do
	for beta in '10';do
		for gamma in '0.15';do
       
            		for dropout in '0.5';do
            
                		for wd in 0.000001;do
					for up_mult in '3' '4' '5'; do
						for dw_mult in '3' '4' '5';do
                					CUDA_VISIBLE_DEVICES=2 python tools/train.py --alpha ${alpha} --beta ${beta} --gamma ${gamma} --dropout ${dropout} --wd ${wd} --up_mult ${up_mult} --dw_mult ${dw_mult} 
                					CUDA_VISIBLE_DEVICES=2 python tools/test.py > "VOC_"$(date +%s)"_${alpha}_${beta}_${gamma}_${dropout}_${wd}_${up_mult}_${dw_mult}_64.txt" 2>&1
						done
					done
				done
	    		done
    		done      
	done
done	
