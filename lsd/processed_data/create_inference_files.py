"""
Script to convert processed datasets to inference-ready format.
The inference format removes assistant responses so LLaMA 3.3 70B can generate them.
"""

import json


def create_instruction_pairs_inference():
    """Convert instruction_pairs to inference format."""
    with open('instruction_pairs_processed.json', encoding='utf-8') as f:
        data = json.load(f)

    inference_data = {
        'dataset_info': {
            'name': 'AILiarDataset (inference-ready)',
            'description': 'Instruction pairs formatted for model completion - NO assistant response included',
            'format': 'Messages ready for LLaMA 3.3 70B inference',
            'source': 'How to Catch an AI Liar paper (Appendix F)',
            'url': 'https://arxiv.org/abs/2309.15840'
        },
        'num_examples': data['num_examples'],
        'data': []
    }

    for item in data['data']:
        # Extract assistant prefix from the response (e.g., 'AI:', 'Salesperson:', etc.)
        assistant_prefix = ''
        if item.get('assistant_response'):
            response = item['assistant_response']
            if ':' in response:
                assistant_prefix = response.split(':')[0] + ':'
        
        inference_item = {
            'id': item['id'],
            'messages': [
                {'role': 'system', 'content': item['system_instruction']},
                {'role': 'user', 'content': item['user_message']}
            ],
            'assistant_prefix': assistant_prefix,
            'label': item['label'],
            'metadata': {
                'pair_id': item.get('pair_id'),
                'variant': item.get('variant'),
                'expected_response': item.get('assistant_response')
            }
        }
        inference_data['data'].append(inference_item)

    with open('instruction_pairs_inference.json', 'w', encoding='utf-8') as f:
        json.dump(inference_data, f, indent=2, ensure_ascii=False)

    print(f'Created instruction_pairs_inference.json with {len(inference_data["data"])} examples')


def create_roleplaying_inference():
    """Convert roleplaying to inference format."""
    with open('roleplaying_processed.json', encoding='utf-8') as f:
        data = json.load(f)

    inference_data = {
        'dataset_info': {
            'name': 'RolePlayingDataset (inference-ready)',
            'description': 'Roleplay scenarios formatted for model completion - NO assistant response included',
            'format': 'Messages ready for LLaMA 3.3 70B inference',
            'source': 'Custom scenarios from deception-detection repo'
        },
        'num_examples': data['num_examples'],
        'data': []
    }

    for item in data['data']:
        # Extract messages without assistant role
        messages = []
        assistant_content = None
        assistant_prefix = ''
        
        for msg in item.get('dialogue', []):
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                # Extract prefix (e.g., "Student:", "Salesperson:")
                if ':' in msg['content']:
                    assistant_prefix = msg['content'].split(':')[0] + ':'
            else:
                messages.append({'role': msg['role'], 'content': msg['content']})
        
        inference_item = {
            'id': item['id'],
            'messages': messages,
            'assistant_prefix': assistant_prefix,
            'label': item['label'],
            'metadata': {
                'pair_id': item.get('pair_id'),
                'variant': item.get('variant'),
                'scenario': item.get('scenario'),
                'expected_response': assistant_content
            }
        }
        inference_data['data'].append(inference_item)

    with open('roleplaying_inference.json', 'w', encoding='utf-8') as f:
        json.dump(inference_data, f, indent=2, ensure_ascii=False)

    print(f'Created roleplaying_inference.json with {len(inference_data["data"])} examples')


def update_summary():
    """Update summary.json with new inference files."""
    with open('summary.json', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Add inference files info
    summary['datasets']['instruction_pairs_inference'] = {
        'file': 'instruction_pairs_inference.json',
        'info': {
            'name': 'AILiarDataset (inference-ready)',
            'description': 'Formatted for LLaMA 3.3 70B inference - no assistant response',
            'format': 'Messages array with assistant_prefix for completion'
        }
    }
    
    summary['datasets']['roleplaying_inference'] = {
        'file': 'roleplaying_inference.json',
        'info': {
            'name': 'RolePlayingDataset (inference-ready)',
            'description': 'Formatted for LLaMA 3.3 70B inference - no assistant response',
            'format': 'Messages array with assistant_prefix for completion'
        }
    }
    
    with open('summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print('Updated summary.json')


if __name__ == '__main__':
    create_instruction_pairs_inference()
    create_roleplaying_inference()
    update_summary()
    print('\nDone! Inference-ready files created.')

