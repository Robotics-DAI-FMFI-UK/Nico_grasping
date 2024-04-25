import json
data = {}
for grasp_type in 'power','precision','side':
    sequence = {}
    for i in range(1, 11):
        file_name = str(i) + ".txt"
        try:
            with open(grasp_type+'/'+file_name, 'r') as file:
                content = file.read()
                grasp = {}
                for j, line in enumerate(content.split('\n')):
                    if line:
                        grasp[f'position {j}'] = [float(j) for j in line.split(',')]
                sequence[f'sequence {i-1}'] = grasp
                        
        except FileNotFoundError:
            print(f"Súbor {file_name} nebol nájdený.")
        data[grasp_type] = sequence
        

formatted_json = json.dumps(data, indent=1)

with open('dataset.json', 'w') as file:
    file.write(formatted_json)

