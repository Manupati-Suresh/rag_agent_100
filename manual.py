import asyncio
from rag_agent_telegram import create_telegram_rag_agent

async def sync_documents():
    print("Initializing RAG agent with Telegram storage...")
    agent = await create_telegram_rag_agent(use_telegram=True)
    
    print("\nSyncing documents from Telegram...")
    await agent.sync_from_telegram()
    
    print("\nCurrent documents in storage:")
    stats = await agent.get_storage_stats()
    print(stats)
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(sync_documents())