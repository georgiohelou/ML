def process_inverted():  
    with open('test2.txt') as f:
        lines = f.readlines()
        Dict = {}
        for i in range(len(lines)):
            current_line=lines[i]
            key=int(current_line.split('----')[0])
            
            abstract_size=current_line.split('\"IndexLength\":')[1]
            abstract_size=int(abstract_size.split(",")[0])
            abstract_array=[None]*abstract_size
            current_line=current_line[:-2]
            mid_part=current_line.split('{')[2]

            if(len(current_line.split('{'))>3):
                for z in range(3,len(current_line.split('{'))):
                    mid_part=mid_part+current_line.split('{')[z]
            pattern = "\"(.*?)\""


            #mid_part=mid_part.split('}')[0]

            instances=mid_part.split('],')

            for j in range(len(instances)):
                current_instance=instances[j]
                print(current_instance)
                word=current_instance.split('\"')[1]
                #word=word.replace('.','')
                current_instance = current_instance.replace(']','')
                current_instance=current_instance+","
                current_instance=current_instance.split("[")[1]
                current_instance=current_instance.split("]")[0]

                numbers_word=current_instance.split(",")
                
                for k in range(len(numbers_word)-1):
                    if(numbers_word[k].isdigit()):
                        abstract_array[int(numbers_word[k])]=word
            print(key)
            abstract_string=""
            for l in range(len(abstract_array)):
                if(not(abstract_array[l]==None)):
                    abstract_string=abstract_string+" "+abstract_array[l]
            
            Dict[key]=abstract_string
        
        print(len(Dict))
            